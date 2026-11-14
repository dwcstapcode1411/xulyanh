import os
import sys
import warnings

# 1. Tắt triệt để các cảnh báo rác trên Python 3.14
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from taxonomy import taxonomy_tree, build_graph_structure
from dataset import build_matador_dataloaders
from models import FeatureExtractor, HierarchicalGATModel, initialize_node_features
from losses import GreedyHierarchicalLoss, TreeDistanceLoss, FocalHierarchicalLoss
import evaluate

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def evaluate_val(feature_extractor, gat_model, dataloader, criterion, level_nodes_map, device):
    feature_extractor.eval()
    gat_model.eval()

    total_loss = 0.0
    correct_by_level = {i: 0 for i in range(5)}
    total_by_level = {i: 0 for i in range(5)}

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            gpu_targets = {
                "multi_hot": targets["multi_hot"].to(device),
                "level_indices": targets["level_indices"].to(device)
            }

            with torch.amp.autocast('cuda'):
                phi_x = feature_extractor(images)
                logits = gat_model(phi_x)
                loss = criterion(logits, gpu_targets, level_nodes_map)

            total_loss += loss.item() * images.size(0)

            # Tính Top-1 Accuracy cho từng tầng phân cấp (Level 0 đến 4)
            level_indices = targets["level_indices"].to(device)

            for d in range(5):
                if d in level_nodes_map:
                    cand_nodes = level_nodes_map[d].to(device)
                    sub_logits = logits[:, cand_nodes]
                    best_cand_idx = torch.argmax(sub_logits, dim=-1)
                    pred_node_ids = cand_nodes[best_cand_idx]
                else:
                    pred_node_ids = torch.argmax(logits, dim=-1)

                target_ids = level_indices[:, d]
                mask = target_ids >= 0
                if mask.sum() > 0:
                    correct = (pred_node_ids[mask] == target_ids[mask]).sum().item()
                    correct_by_level[d] += correct
                    total_by_level[d] += mask.sum().item()

    avg_loss = total_loss / len(dataloader.dataset)
    acc_by_level = {}
    for d in range(5):
        acc_by_level[d] = (correct_by_level[d] / total_by_level[d] * 100.0) if total_by_level[d] > 0 else 0.0

    return avg_loss, acc_by_level

def main():
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình Matador-C1 Hierarchical GAT trên ảnh thật")
    parser.add_argument("--epochs", type=int, default=25, help="Số epochs huấn luyện (chuẩn 25 epochs)")
    parser.add_argument("--batch_size", type=int, default=32, help="Kích thước batch (khuyến nghị 32 cho RTX 4060)")
    parser.add_argument("--lr_backbone", type=float, default=1e-4, help="Learning rate của backbone ResNet50")
    parser.add_argument("--lr_gat", type=float, default=1e-3, help="Learning rate của GAT head")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay chuẩn cho AdamW")
    parser.add_argument("--feature_dim", type=int, default=256, help="Số chiều vector đặc trưng")
    parser.add_argument("--mode", type=str, default="c1", choices=["c1", "full"], help="Cấu hình Matador-C1 hoặc Full")
    parser.add_argument("--loss", type=str, default="greedy", choices=["greedy", "tree_dist", "focal"], help="Hàm mất mát: greedy (chuẩn), tree_dist (phạt khoảng cách cây), focal (tập trung mẫu khó)")
    parser.add_argument("--tree_dist_alpha", type=float, default=0.3, help="Hệ số phạt khoảng cách cây alpha cho tree_dist loss")
    parser.add_argument("--init_checkpoint", type=str, default=None, help="Đường dẫn checkpoint để nạp trọng số khởi đầu (fine-tune)")
    parser.add_argument("--num_workers", type=int, default=2, help="Số luồng nạp dữ liệu")
    parser.add_argument("--resume", action="store_true", help="Tiếp tục từ checkpoint trước nếu có")
    parser.add_argument("--seed", type=int, default=42, help="Seed ngẫu nhiên")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== KHỞI CHẠY HUẤN LUYỆN MATADOR-C1 ({args.epochs} EPOCHS) TRÊN THIẾT BỊ: {device} ===")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    # 1. Nạp tập dữ liệu thật Matador-C1 (lược bỏ 5 lớp phẳng, còn 37 lớp, ~6.600 mẫu)
    print("-> Đang nạp và phân tầng bộ dữ liệu chuẩn Matador-C1...")
    train_loader, val_loader, test_loader, graph_meta = build_matador_dataloaders(
        data_dir="matador",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        tree=taxonomy_tree,
        mode=args.mode,
        seed=args.seed
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
        level_nodes_map[lvl] = torch.tensor(sorted(level_nodes_map[lvl]), dtype=torch.long)

    # 2. Khởi tạo mô hình FeatureExtractor & HierarchicalGATModel (nâng cấp chống over-squashing)
    feature_extractor = FeatureExtractor(
        model_name='resnet50',
        pretrained=True,
        feature_dim=args.feature_dim
    ).to(device)

    gat_model = HierarchicalGATModel(
        num_nodes=num_nodes,
        edge_index=edge_index,
        global_node_idx=global_node_idx,
        feature_dim=args.feature_dim,
        num_layers=3
    ).to(device)

    # 3. Khởi tạo vector đặc trưng nút h_i^0 hoặc nạp từ checkpoint khởi đầu
    if args.init_checkpoint and os.path.isfile(args.init_checkpoint):
        print(f"-> Nạp trọng số khởi đầu để tinh chỉnh (Fine-tune) từ: {args.init_checkpoint}")
        init_ckpt = torch.load(args.init_checkpoint, map_location=device)
        feature_extractor.load_state_dict(init_ckpt['feature_extractor_state'])
        gat_model.load_state_dict(init_ckpt['gat_model_state'])
    else:
        initial_h0 = initialize_node_features(
            feature_extractor,
            train_loader,
            nodes,
            node_to_idx,
            node_levels,
            device
        )
        with torch.no_grad():
            gat_model.node_embeddings.copy_(initial_h0.to(device))

    # 4. Thiết lập Optimizer AdamW với weight decay 1e-2, CosineAnnealingLR và GreedyHierarchicalLoss
    optimizer = optim.AdamW([
        {'params': feature_extractor.parameters(), 'lr': args.lr_backbone},
        {'params': gat_model.parameters(), 'lr': args.lr_gat}
    ], weight_decay=args.weight_decay)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    if args.loss == "tree_dist":
        print(f"-> Sử dụng TreeDistanceLoss (alpha={args.tree_dist_alpha}) phạt lỗi sai theo khoảng cách phân cấp trên cây.")
        criterion = TreeDistanceLoss(nodes, node_to_idx, node_levels, taxonomy_tree, alpha=args.tree_dist_alpha, label_smoothing=0.05).to(device)
    elif args.loss == "focal":
        print("-> Sử dụng FocalHierarchicalLoss tập trung vào các mẫu khó phân loại ở tầng lá.")
        criterion = FocalHierarchicalLoss(node_levels=node_levels, label_smoothing=0.05).to(device)
    else:
        print("-> Sử dụng GreedyHierarchicalLoss chuẩn theo bài báo ICCV 2025.")
        criterion = GreedyHierarchicalLoss(node_levels=node_levels, label_smoothing=0.05).to(device)
    scaler = torch.amp.GradScaler('cuda')

    start_epoch = 0
    best_val_acc = -1.0
    best_val_loss = float('inf')

    checkpoint_last_path = os.path.join("checkpoints", "last_checkpoint.pth")
    checkpoint_best_path = os.path.join("checkpoints", "best_hierarchical_model.pth")

    if args.resume and os.path.isfile(checkpoint_last_path):
        print(f"-> Đang nạp checkpoint tiếp tục từ: {checkpoint_last_path}")
        ckpt = torch.load(checkpoint_last_path, map_location=device)
        feature_extractor.load_state_dict(ckpt['feature_extractor_state'])
        gat_model.load_state_dict(ckpt['gat_model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        if 'scaler_state' in ckpt:
            scaler.load_state_dict(ckpt['scaler_state'])
        start_epoch = ckpt.get('epoch', 0) + 1
        best_val_acc = ckpt.get('best_val_acc', -1.0)
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        print(f"-> Tiếp tục từ Epoch {start_epoch + 1}/{args.epochs} (Best Val Acc: {best_val_acc:.2f}%)")

    # 5. Vòng lặp huấn luyện End-to-End 25 Epochs
    print(f"\n===== BẮT ĐẦU HUẤN LUYỆN MATADOR-C1 ({args.epochs} EPOCHS) =====")

    for epoch in range(start_epoch, args.epochs):
        feature_extractor.train()
        gat_model.train()

        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [Train]", unit="batch")

        for images, targets in pbar:
            images = images.to(device)
            gpu_targets = {
                "multi_hot": targets["multi_hot"].to(device),
                "level_indices": targets["level_indices"].to(device)
            }

            optimizer.zero_grad()

            with torch.amp.autocast('cuda'):
                phi_x = feature_extractor(images)
                logits = gat_model(phi_x)
                loss = criterion(logits, gpu_targets, level_nodes_map)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * images.size(0)
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        scheduler.step()
        epoch_train_loss = running_loss / len(train_loader.dataset)

        # Đánh giá trên tập Val
        val_loss, val_acc_by_level = evaluate_val(
            feature_extractor, gat_model, val_loader, criterion, level_nodes_map, device
        )
        leaf_acc = val_acc_by_level[4] # Độ chính xác tầng lá Material (Level 4)

        print(f"\n[Epoch {epoch + 1}/{args.epochs}] Train Loss: {epoch_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Leaf Acc: {leaf_acc:.2f}%")
        print(f"  -> Chi tiết Acc từng tầng: Phase={val_acc_by_level[0]:.1f}%, State={val_acc_by_level[1]:.1f}%, Comp={val_acc_by_level[2]:.1f}%, Form={val_acc_by_level[3]:.1f}%, Material={val_acc_by_level[4]:.1f}%")

        # Lưu checkpoint hiện tại
        checkpoint_dict = {
            'epoch': epoch,
            'feature_extractor_state': feature_extractor.state_dict(),
            'gat_model_state': gat_model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scaler_state': scaler.state_dict(),
            'val_loss': val_loss,
            'val_acc_by_level': val_acc_by_level,
            'best_val_acc': best_val_acc,
            'best_val_loss': best_val_loss,
            'graph_meta': graph_meta,
            'mode': args.mode
        }
        torch.save(checkpoint_dict, checkpoint_last_path)

        # Cập nhật checkpoint tốt nhất khi Leaf Acc cao hơn
        if leaf_acc > best_val_acc or (leaf_acc == best_val_acc and val_loss < best_val_loss):
            best_val_acc = leaf_acc
            best_val_loss = val_loss
            checkpoint_dict['best_val_acc'] = best_val_acc
            checkpoint_dict['best_val_loss'] = best_val_loss
            torch.save(checkpoint_dict, checkpoint_best_path)
            print(f"   >>> [*] Đã lưu checkpoint tốt nhất vào {checkpoint_best_path} (Val Leaf Acc: {best_val_acc:.2f}%)!")

    print(f"\n===== HUẤN LUYỆN 25 EPOCHS HOÀN TẤT =====")
    print(f"Trọng số tối ưu nhất lưu tại: {checkpoint_best_path}")

    # 6. Tự động chạy evaluate.py trên tập Test và xuất biểu đồ
    print("\n-> Bắt đầu tự động đánh giá mô hình trên tập Test thật...")
    evaluate.run_evaluation(
        checkpoint_path=checkpoint_best_path,
        data_dir="matador",
        mode=args.mode
    )

if __name__ == "__main__":
    main()
