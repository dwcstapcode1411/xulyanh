"""
test_dataset.py - Kiểm thử module dataset (quét dữ liệu, phân chia stratified, chuyển đổi ảnh TIFF).
"""
import sys
import os
import unittest
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PIL import Image
import torch

from taxonomy import MATADOR_C1_TREE, OMITTED_CLASSES, CONSOLIDATION_MAP, build_graph_structure
from dataset import scan_matador_dataset, get_stratified_split, get_transforms, MatadorRealDataset


DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "matador")
SKIP_MSG = f"Matador dataset not found at {DATA_DIR}"


class TestScanMatadorDataset(unittest.TestCase):
    """Kiểm tra quét thư mục dữ liệu Matador."""

    @unittest.skipIf(not os.path.isdir(os.path.join(DATA_DIR, "texture_img")), SKIP_MSG)
    def test_c1_scan_returns_samples(self):
        """Quét mode='c1' phải trả về mẫu hợp lệ."""
        samples = scan_matador_dataset(DATA_DIR, mode="c1")
        self.assertGreater(len(samples), 0, "No samples found in C1 mode!")
        # Theo bài báo: ~6,600 mẫu
        self.assertGreater(len(samples), 5000, f"Expected ~6600 samples, got {len(samples)}")

    @unittest.skipIf(not os.path.isdir(os.path.join(DATA_DIR, "texture_img")), SKIP_MSG)
    def test_c1_excludes_omitted_classes(self):
        """Mode='c1' phải loại bỏ các lớp trong OMITTED_CLASSES."""
        samples = scan_matador_dataset(DATA_DIR, mode="c1")
        labels = set([s[1] for s in samples])
        for omitted in OMITTED_CLASSES:
            self.assertNotIn(omitted, labels, f"Omitted class '{omitted}' found in C1 samples!")

    @unittest.skipIf(not os.path.isdir(os.path.join(DATA_DIR, "texture_img")), SKIP_MSG)
    def test_c1_labels_match_tree_leaves(self):
        """Mọi nhãn trong C1 phải tồn tại trong danh sách lá của cây C1."""
        samples = scan_matador_dataset(DATA_DIR, mode="c1")
        _, _, node_levels, _ = build_graph_structure(MATADOR_C1_TREE)
        c1_leaves = set(n for n, lvl in node_levels.items() if lvl == 4 and n != "GLOBAL_NODE")
        sample_labels = set(s[1] for s in samples)
        for label in sample_labels:
            self.assertIn(label, c1_leaves,
                          f"Sample label '{label}' not found in C1 tree leaves: {sorted(c1_leaves)}")

    @unittest.skipIf(not os.path.isdir(os.path.join(DATA_DIR, "texture_img")), SKIP_MSG)
    def test_all_image_paths_exist(self):
        """Mọi đường dẫn ảnh trong samples phải tồn tại."""
        samples = scan_matador_dataset(DATA_DIR, mode="c1")
        # Kiểm tra ngẫu nhiên 100 mẫu để tránh quá lâu
        import random
        random.seed(42)
        check_samples = random.sample(samples, min(100, len(samples)))
        for img_path, _ in check_samples:
            self.assertTrue(os.path.isfile(img_path), f"Image file not found: {img_path}")

    @unittest.skipIf(not os.path.isdir(os.path.join(DATA_DIR, "texture_img")), SKIP_MSG)
    def test_full_scan_returns_more_samples(self):
        """Mode='full' phải trả về nhiều mẫu hơn mode='c1' (bao gồm cả lớp bị loại bỏ)."""
        samples_c1 = scan_matador_dataset(DATA_DIR, mode="c1")
        samples_full = scan_matador_dataset(DATA_DIR, mode="full")
        self.assertGreaterEqual(len(samples_full), len(samples_c1),
                                f"Full ({len(samples_full)}) should >= C1 ({len(samples_c1)})")


class TestStratifiedSplit(unittest.TestCase):
    """Kiểm tra phân chia dữ liệu stratified."""

    @unittest.skipIf(not os.path.isdir(os.path.join(DATA_DIR, "texture_img")), SKIP_MSG)
    def test_split_ratios(self):
        """Tỷ lệ phân chia phải xấp xỉ 80/10/10."""
        samples = scan_matador_dataset(DATA_DIR, mode="c1")
        train, val, test = get_stratified_split(samples)
        total = len(samples)

        train_ratio = len(train) / total
        val_ratio = len(val) / total
        test_ratio = len(test) / total

        self.assertAlmostEqual(train_ratio, 0.80, delta=0.03,
                               msg=f"Train ratio {train_ratio:.3f} too far from 0.80")
        self.assertAlmostEqual(val_ratio, 0.10, delta=0.03,
                               msg=f"Val ratio {val_ratio:.3f} too far from 0.10")
        self.assertAlmostEqual(test_ratio, 0.10, delta=0.03,
                               msg=f"Test ratio {test_ratio:.3f} too far from 0.10")

    @unittest.skipIf(not os.path.isdir(os.path.join(DATA_DIR, "texture_img")), SKIP_MSG)
    def test_no_data_leakage(self):
        """Không có dữ liệu bị rò rỉ (trùng lặp) giữa train/val/test."""
        samples = scan_matador_dataset(DATA_DIR, mode="c1")
        train, val, test = get_stratified_split(samples)

        train_paths = set(s[0] for s in train)
        val_paths = set(s[0] for s in val)
        test_paths = set(s[0] for s in test)

        self.assertEqual(len(train_paths & val_paths), 0, "Data leakage between train and val!")
        self.assertEqual(len(train_paths & test_paths), 0, "Data leakage between train and test!")
        self.assertEqual(len(val_paths & test_paths), 0, "Data leakage between val and test!")

    @unittest.skipIf(not os.path.isdir(os.path.join(DATA_DIR, "texture_img")), SKIP_MSG)
    def test_total_preserved(self):
        """Tổng số mẫu sau khi chia phải bằng tổng ban đầu."""
        samples = scan_matador_dataset(DATA_DIR, mode="c1")
        train, val, test = get_stratified_split(samples)
        self.assertEqual(len(train) + len(val) + len(test), len(samples),
                         "Total samples after split != original total!")

    @unittest.skipIf(not os.path.isdir(os.path.join(DATA_DIR, "texture_img")), SKIP_MSG)
    def test_deterministic_split(self):
        """Phân chia phải cho kết quả giống nhau khi dùng cùng seed."""
        samples = scan_matador_dataset(DATA_DIR, mode="c1")
        train1, val1, test1 = get_stratified_split(samples, seed=42)
        train2, val2, test2 = get_stratified_split(samples, seed=42)

        self.assertEqual(len(train1), len(train2))
        self.assertEqual([s[0] for s in test1], [s[0] for s in test2])

    @unittest.skipIf(not os.path.isdir(os.path.join(DATA_DIR, "texture_img")), SKIP_MSG)
    def test_all_classes_in_each_split(self):
        """Mọi lớp phải xuất hiện trong cả 3 phần chia (stratified)."""
        samples = scan_matador_dataset(DATA_DIR, mode="c1")
        train, val, test = get_stratified_split(samples)

        all_labels = set(s[1] for s in samples)
        train_labels = set(s[1] for s in train)
        val_labels = set(s[1] for s in val)
        test_labels = set(s[1] for s in test)

        for label in all_labels:
            self.assertIn(label, train_labels, f"Class '{label}' missing from train split!")
            self.assertIn(label, val_labels, f"Class '{label}' missing from val split!")
            self.assertIn(label, test_labels, f"Class '{label}' missing from test split!")


class TestTransforms(unittest.TestCase):
    """Kiểm tra data transforms."""

    def test_train_transform_output(self):
        """Transform train phải tạo tensor đã normalize."""
        train_tf, _ = get_transforms(img_size=224)
        img = Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8))
        out = train_tf(img)
        self.assertEqual(out.shape, (3, 224, 224))
        self.assertIsInstance(out, torch.Tensor)

    def test_val_transform_output(self):
        """Transform val/test phải tạo tensor kích thước đúng."""
        _, val_tf = get_transforms(img_size=224)
        img = Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8))
        out = val_tf(img)
        self.assertEqual(out.shape, (3, 224, 224))

    def test_custom_img_size(self):
        """Kích thước ảnh tùy chỉnh (384) phải được áp dụng đúng."""
        _, val_tf = get_transforms(img_size=384)
        img = Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8))
        out = val_tf(img)
        self.assertEqual(out.shape, (3, 384, 384))


class TestTIFFReading(unittest.TestCase):
    """Kiểm tra khả năng đọc ảnh TIFF."""

    @unittest.skipIf(not os.path.isdir(os.path.join(DATA_DIR, "texture_img")), SKIP_MSG)
    def test_read_tiff_as_rgb(self):
        """Đọc file TIFF và convert sang RGB phải thành công."""
        import glob
        tiff_files = glob.glob(os.path.join(DATA_DIR, "texture_img", "*.tiff"))
        if not tiff_files:
            self.skipTest("No TIFF files found")

        img = Image.open(tiff_files[0]).convert('RGB')
        arr = np.array(img)
        self.assertEqual(len(arr.shape), 3)
        self.assertEqual(arr.shape[2], 3)
        self.assertGreater(arr.shape[0], 0)
        self.assertGreater(arr.shape[1], 0)


class TestMatadorRealDataset(unittest.TestCase):
    """Kiểm tra MatadorRealDataset class."""

    @unittest.skipIf(not os.path.isdir(os.path.join(DATA_DIR, "texture_img")), SKIP_MSG)
    def test_dataset_getitem(self):
        """__getitem__ phải trả về tuple (image_tensor, target_dict)."""
        samples = scan_matador_dataset(DATA_DIR, mode="c1")
        nodes, node_to_idx, _, _ = build_graph_structure(MATADOR_C1_TREE)
        _, val_tf = get_transforms(img_size=224)
        dataset = MatadorRealDataset(samples[:10], node_to_idx, tree=MATADOR_C1_TREE, transform=val_tf)

        img, target = dataset[0]
        self.assertEqual(img.shape, (3, 224, 224))
        self.assertIn("multi_hot", target)
        self.assertIn("level_indices", target)
        self.assertIn("mechanical_labels", target)
        self.assertIn("leaf_name", target)
        self.assertEqual(target["level_indices"].shape, (5,))
        self.assertEqual(target["mechanical_labels"].shape, (3,))

    @unittest.skipIf(not os.path.isdir(os.path.join(DATA_DIR, "texture_img")), SKIP_MSG)
    def test_multi_hot_label_correctness(self):
        """Nhãn multi-hot phải có đúng 5 vị trí = 1 (tương ứng 5 tầng phân cấp)."""
        samples = scan_matador_dataset(DATA_DIR, mode="c1")
        nodes, node_to_idx, _, _ = build_graph_structure(MATADOR_C1_TREE)
        _, val_tf = get_transforms(img_size=224)
        dataset = MatadorRealDataset(samples[:50], node_to_idx, tree=MATADOR_C1_TREE, transform=val_tf)

        for i in range(min(50, len(dataset))):
            _, target = dataset[i]
            num_hot = target["multi_hot"].sum().item()
            self.assertEqual(num_hot, 5.0,
                             f"Sample {i} has {num_hot} hot labels (expected 5)")


if __name__ == "__main__":
    unittest.main(verbosity=2)
