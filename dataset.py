import os
import sys
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import glob
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

from taxonomy import (
    taxonomy_tree,
    MATADOR_C1_TREE,
    OMITTED_CLASSES,
    CONSOLIDATION_MAP,
    find_taxonomy_path,
    get_binary_mechanical_labels,
    build_graph_structure
)

class MatadorRealDataset(Dataset):
    """
    Dataset nạp ảnh thật (.tiff) và nhãn thật (.txt) từ thư mục Matador.
    """
    def __init__(self, sample_list, node_to_idx, tree=taxonomy_tree, transform=None):
        self.sample_list = sample_list
        self.node_to_idx = node_to_idx
        self.tree = tree
        self.transform = transform
        self.num_nodes = len(node_to_idx)

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        img_path, raw_material = self.sample_list[idx]

        # Ánh xạ nhãn sang nhóm Matador-C1
        leaf_material = CONSOLIDATION_MAP.get(raw_material.lower(), raw_material.lower())

        # Đọc ảnh từ file TIFF (RGB)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        # 1. Tìm đường dẫn phân cấp (Phase -> State -> Composition -> Form -> Material)
        path = find_taxonomy_path(self.tree, leaf_material)

        # 2. Tạo nhãn multi-hot và chỉ số node_id cho từng tầng
        multi_hot = torch.zeros(self.num_nodes, dtype=torch.float32)
        level_indices = -torch.ones(5, dtype=torch.long)

        if path:
            for level, node_name in enumerate(path):
                if node_name in self.node_to_idx:
                    node_id = self.node_to_idx[node_name]
                    multi_hot[node_id] = 1.0
                    if level < 5:
                        level_indices[level] = node_id

        # 3. Lấy nhãn cơ tính nhị phân [Strong, Rigid, Heavy]
        mech_labels = torch.tensor(get_binary_mechanical_labels(leaf_material), dtype=torch.float32)

        return image, {
            "multi_hot": multi_hot,
            "level_indices": level_indices,
            "mechanical_labels": mech_labels,
            "leaf_name": leaf_material,
            "img_path": img_path
        }

def get_transforms(img_size=224):
    """
    Data Augmentation chuẩn ImageNet cho huấn luyện và kiểm thử
    """
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.75, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_test_transform

def scan_matador_dataset(data_dir="matador", mode="c1"):
    """
    Quét thư mục data_dir và lọc theo cấu hình:
    mode='c1': Loại bỏ 5 lớp vật liệu phẳng (glass, paint, thermoplastic, thermoset, elastomer)
               và gộp các kim loại, gốm, đất... về 37 lớp chuẩn theo Section S2.5.
    """
    img_dir = os.path.join(data_dir, "texture_img")
    label_dir = os.path.join(data_dir, "label")

    if not os.path.isdir(img_dir) or not os.path.isdir(label_dir):
        raise FileNotFoundError(f"Không tìm thấy thư mục {img_dir} hoặc {label_dir}")

    label_files = glob.glob(os.path.join(label_dir, "*.txt"))
    valid_samples = []

    for l_path in label_files:
        base_name = os.path.splitext(os.path.basename(l_path))[0]
        img_path = os.path.join(img_dir, f"{base_name}.tiff")
        if os.path.isfile(img_path):
            with open(l_path, "r", encoding="utf-8") as f:
                label_str = f.read().strip().lower()

            if not label_str:
                continue

            if mode == "c1":
                # Lược bỏ các vật liệu phẳng không có kết cấu
                if label_str in OMITTED_CLASSES:
                    continue
                # Ánh xạ sang tên lớp gộp
                mapped_label = CONSOLIDATION_MAP.get(label_str, label_str)
                valid_samples.append((img_path, mapped_label))
            else:
                valid_samples.append((img_path, label_str))

    return valid_samples

def get_stratified_split(samples, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    labels = [s[1] for s in samples]

    train_samples, temp_samples = train_test_split(
        samples,
        test_size=(val_ratio + test_ratio),
        stratify=labels,
        random_state=seed
    )

    temp_labels = [s[1] for s in temp_samples]
    val_samples, test_samples = train_test_split(
        temp_samples,
        test_size=0.5,
        stratify=temp_labels,
        random_state=seed
    )

    return train_samples, val_samples, test_samples

def build_matador_dataloaders(data_dir="matador", batch_size=32, num_workers=2, tree=taxonomy_tree, mode="c1", seed=42, img_size=224):
    """
    Quét và tạo DataLoader cho Matador (Mặc định Matador-C1: 37 lớp, ~6.600 mẫu)
    """
    samples = scan_matador_dataset(data_dir, mode=mode)
    print(f"Tổng số mẫu ảnh thật được nạp thành công ({mode.upper()}): {len(samples)}")

    train_samples, val_samples, test_samples = get_stratified_split(samples, seed=seed)
    print(f"Số lượng: Train={len(train_samples)} (80%), Val={len(val_samples)} (10%), Test={len(test_samples)} (10%)")

    nodes, node_to_idx, node_levels, edge_index = build_graph_structure(tree)
    train_transform, val_test_transform = get_transforms(img_size=img_size)

    train_dataset = MatadorRealDataset(train_samples, node_to_idx, tree=tree, transform=train_transform)
    val_dataset = MatadorRealDataset(val_samples, node_to_idx, tree=tree, transform=val_test_transform)
    test_dataset = MatadorRealDataset(test_samples, node_to_idx, tree=tree, transform=val_test_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    graph_meta = {
        "nodes": nodes,
        "node_to_idx": node_to_idx,
        "node_levels": node_levels,
        "edge_index": edge_index,
        "global_node_idx": node_to_idx["GLOBAL_NODE"]
    }

    return train_loader, val_loader, test_loader, graph_meta
