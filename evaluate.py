import os
import sys
import warnings

# 1. Tắt cảnh báo rác trên Python 3.14
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import torch
import torch.nn.functional as F

from taxonomy import taxonomy_tree, build_graph_structure, find_taxonomy_path
from dataset import build_matador_dataloaders
from models import FeatureExtractor, HierarchicalGATModel

def build_leaf_to_parent_form_map(tree, node_levels, node_to_idx):
    """
    Xây dựng bảng tra cứu: từ chỉ số nút lá Material (Level 4) -> chỉ số nút cha Form (Level 3).
    """
    leaf_to_parent_form = {}
    for node_name, lvl in node_levels.items():
        if lvl == 4 and node_name != "GLOBAL_NODE":
            path = find_taxonomy_path(tree, node_name)
            if path and len(path) == 5:
                form_name = path[3]
                if form_name in node_to_idx and node_name in node_to_idx:
                    leaf_to_parent_form[node_to_idx[node_name]] = node_to_idx[form_name]
    return leaf_to_parent_form

@torch.no_grad()
def evaluate_dataloader(feature_extractor, gat_model, dataloader, level_nodes_map, leaf_to_parent_form, device):
    """
    Đánh giá mô hình theo chuẩn Single-view (Resize 224x224):
    - Level 0 (Phase), Level 1 (State), Level 2 (Composition), Level 4 (Material): argmax chuẩn.
    - Level 3 (Form): Suy ra trực tiếp từ nút cha của nhãn Material dự đoán.
    """
    feature_extractor.eval()
    gat_model.eval()

    correct_by_level = {i: 0 for i in range(5)}
    total_by_level = {i: 0 for i in range(5)}
    all_preds_leaf = []
    all_targets_leaf = []

    for images, targets in dataloader:
        images = images.to(device)
        with torch.amp.autocast('cuda'):
            phi_x = feature_extractor(images)
            logits = gat_model(phi_x)

        level_indices = targets["level_indices"].to(device) # (B, 5)

        # 1. Dự đoán độc lập cho các tầng
        preds = {}
        for d in range(5):
            cand = level_nodes_map[d]
            preds[d] = cand[torch.argmax(logits[:, cand], dim=-1)]

        # 2. Suy ra nhãn Form (Level 3) từ nút cha trực tiếp của Material (Level 4)
        pred_mats = preds[4].cpu().numpy()
        form_deduced = []
        for i, m_id in enumerate(pred_mats):
            form_id = leaf_to_parent_form.get(m_id, preds[3][i].item())
            form_deduced.append(form_id)
        preds[3] = torch.tensor(form_deduced, dtype=torch.long, device=device)

        # 3. Tính độ chính xác từng tầng
        for d in range(5):
            gt = level_indices[:, d]
            mask = gt >= 0
            if mask.sum() > 0:
                correct = (preds[d][mask] == gt[mask]).sum().item()
                correct_by_level[d] += correct
                total_by_level[d] += mask.sum().item()

                if d == 4:
                    all_preds_leaf.extend(preds[4][mask].cpu().numpy())
                    all_targets_leaf.extend(gt[mask].cpu().numpy())

    acc_by_level = {}
    for d in range(5):
        acc_by_level[d] = (correct_by_level[d] / total_by_level[d] * 100.0) if total_by_level[d] > 0 else 0.0

    return acc_by_level, all_preds_leaf, all_targets_leaf

def plot_figure_5b_and_cm(acc_by_level, all_preds_leaf, all_targets_leaf, nodes, output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)
    level_names = [
        "Level 0 (Phase)",
        "Level 1 (State)",
        "Level 2 (Composition)",
        "Level 3 (Form)",
        "Level 4 (Material)"
    ]
    accuracies = [acc_by_level.get(i, 0.0) for i in range(5)]

    # 1. Biểu đồ Top-1 Accuracy theo từng tầng phân cấp (Hình 5b của bài báo ICCV 2025)
    plt.figure(figsize=(9, 5))
    palette = sns.color_palette("viridis", len(level_names))
    bars = plt.barh(level_names, accuracies, color=palette)
    plt.xlim(0, 105)
    plt.xlabel("Độ chính xác Top-1 Accuracy (%)", fontsize=12, fontweight='bold')
    plt.title("Độ chính xác Top-1 theo từng cấp độ phân cấp (Hình 5b - Matador-C1)", fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()

    for bar, acc in zip(bars, accuracies):
        plt.text(
            bar.get_width() + 1.2,
            bar.get_y() + bar.get_height() / 2,
            f"{acc:.2f}%",
            va='center',
            ha='left',
            fontweight='bold',
            fontsize=11
        )

    plt.tight_layout()
    fig5b_path = os.path.join(output_dir, "figure_5b_hierarchy_accuracy.png")
    plt.savefig(fig5b_path, dpi=300)
    plt.close()
    print(f"-> Đã cập nhật biểu đồ Hình 5b vào: {fig5b_path}")

    # 2. Confusion Matrix cho các lớp vật liệu ở nút lá
    if all_preds_leaf and all_targets_leaf:
        unique_classes = sorted(list(set(all_targets_leaf + all_preds_leaf)))
        cm = confusion_matrix(all_targets_leaf, all_preds_leaf, labels=unique_classes)
        unique_class_names = [nodes[idx] for idx in unique_classes]

        plt.figure(figsize=(15, 13))
        sns.heatmap(
            cm,
            annot=True if len(unique_classes) <= 37 else False,
            fmt='d',
            cmap='Blues',
            xticklabels=unique_class_names,
            yticklabels=unique_class_names
        )
        plt.title("Ma trận nhầm lẫn phân loại vật liệu nút lá Matador-C1 (Test set)", fontsize=14, fontweight='bold')
        plt.xlabel("Lớp dự đoán (Predicted)", fontsize=12, fontweight='bold')
        plt.ylabel("Lớp thực tế (Ground Truth)", fontsize=12, fontweight='bold')
        plt.xticks(rotation=90, fontsize=8)
        plt.yticks(rotation=0, fontsize=8)
        plt.tight_layout()

        cm_path = os.path.join(output_dir, "confusion_matrix.png")
        plt.savefig(cm_path, dpi=300)
        plt.close()
        print(f"-> Đã cập nhật Confusion Matrix vào: {cm_path}")

def plot_ablation_comparison(baseline_accs, new_accs, output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)
    levels = ["Phase (L0)", "State (L1)", "Composition (L2)", "Form (L3)", "Material (L4)"]
    x = np.arange(len(levels))
    width = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, baseline_accs, width, label='Mô hình cơ sở độc lập (Independent Argmax)', color='#4A90E2')
    plt.bar(x + width/2, new_accs, width, label='Suy luận Nhất quán Phân cấp (Parent-Form Consistent)', color='#50E3C2')

    plt.ylabel('Top-1 Accuracy (%)', fontsize=12, fontweight='bold')
    plt.title('So sánh Ablation Study: Khắc phục Nghịch đảo Tầng Form < Material', fontsize=14, fontweight='bold')
    plt.xticks(x, levels, fontsize=11, fontweight='bold')
    plt.ylim(0, 105)
    plt.legend(fontsize=11)
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    for i in range(len(levels)):
        plt.text(x[i] - width/2, baseline_accs[i] + 1.5, f"{baseline_accs[i]:.2f}%", ha='center', fontsize=9, fontweight='bold')
        plt.text(x[i] + width/2, new_accs[i] + 1.5, f"{new_accs[i]:.2f}%", ha='center', fontsize=9, fontweight='bold', color='#00785A')

    plt.tight_layout()
    path = os.path.join(output_dir, "ablation_comparison.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"-> Đã lưu biểu đồ so sánh Ablation Study vào: {path}")

def run_evaluation(checkpoint_path="checkpoints/best_hierarchical_model.pth", data_dir="matador", mode="c1"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=== ĐÁNH GIÁ MÔ HÌNH TRÊN TẬP TEST THẬT ({mode.upper()}, Single-View 224x224, {device}) ===")

    if not os.path.isfile(checkpoint_path):
        print(f"Lỗi: Không tìm thấy file checkpoint {checkpoint_path}")
        return

    print(f"-> Nạp checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)

    # Nạp tập dữ liệu test qua DataLoader chuẩn
    _, _, test_loader, graph_meta = build_matador_dataloaders(
        data_dir=data_dir,
        batch_size=32,
        num_workers=2,
        tree=taxonomy_tree,
        mode=mode,
        img_size=224
    )

    nodes = graph_meta["nodes"]
    node_to_idx = graph_meta["node_to_idx"]
    node_levels = graph_meta["node_levels"]
    edge_index = graph_meta["edge_index"].to(device)
    num_nodes = len(nodes)
    global_node_idx = graph_meta["global_node_idx"]

    level_nodes_map = {}
    for node_name, lvl in node_levels.items():
        if lvl >= 0 and node_name != "GLOBAL_NODE":
            if lvl not in level_nodes_map:
                level_nodes_map[lvl] = []
            level_nodes_map[lvl].append(node_to_idx[node_name])
    for lvl in level_nodes_map:
        level_nodes_map[lvl] = torch.tensor(sorted(level_nodes_map[lvl]), dtype=torch.long, device=device)

    leaf_to_parent_form = build_leaf_to_parent_form_map(taxonomy_tree, node_levels, node_to_idx)

    feature_extractor = FeatureExtractor(model_name='resnet50', pretrained=False, feature_dim=256).to(device)
    gat_model = HierarchicalGATModel(
        num_nodes=num_nodes,
        edge_index=edge_index,
        global_node_idx=global_node_idx,
        feature_dim=256,
        num_layers=3
    ).to(device)

    feature_extractor.load_state_dict(ckpt['feature_extractor_state'])
    gat_model.load_state_dict(ckpt['gat_model_state'])
    print("-> Đã nạp thành công toàn bộ trọng số mạng Feature Extractor và Hierarchical GAT!")

    # Tiến hành đánh giá
    acc_by_level, preds, targets = evaluate_dataloader(
        feature_extractor, gat_model, test_loader, level_nodes_map, leaf_to_parent_form, device
    )

    print("\n" + "="*65)
    print("===== KẾT QUẢ ĐÁNH GIÁ TOP-1 ACCURACY TRÊN TẬP TEST THẬT (C1) =====")
    print("="*65)
    print(f"Level 0 (Phase)      : {acc_by_level[0]:.2f}%")
    print(f"Level 1 (State)      : {acc_by_level[1]:.2f}%")
    print(f"Level 2 (Composition): {acc_by_level[2]:.2f}%")
    print(f"Level 3 (Form)       : {acc_by_level[3]:.2f}%")
    print(f"Level 4 (Material)   : {acc_by_level[4]:.2f}%")
    print("="*65)

    plot_figure_5b_and_cm(acc_by_level, preds, targets, nodes, output_dir="outputs")
    
    # Baseline comparison (Form trước khi suy luận từ nút cha Material: 81.42%)
    baseline_accs = [100.0, 95.92, 90.48, 81.42, 84.14]
    new_accs = [acc_by_level[i] for i in range(5)]
    plot_ablation_comparison(baseline_accs, new_accs, output_dir="outputs")
    
    return acc_by_level

def main():
    parser = argparse.ArgumentParser(description="Đánh giá mô hình Matador-C1 trên tập Test thật")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_hierarchical_model.pth")
    parser.add_argument("--data_dir", type=str, default="matador")
    parser.add_argument("--mode", type=str, default="c1", choices=["c1", "full"])
    args = parser.parse_args()

    run_evaluation(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        mode=args.mode
    )

if __name__ == "__main__":
    main()
