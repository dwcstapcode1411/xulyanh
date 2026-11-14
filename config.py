"""
config.py - Cấu hình tập trung cho toàn bộ hệ thống Matador Project.
Hỗ trợ định nghĩa cấu hình cho dữ liệu, kiến trúc mô hình, huấn luyện, suy luận và giao diện.
"""
import os

# 1. Đường dẫn thư mục
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "matador")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
DEMO_DIR = os.path.join(BASE_DIR, "demo_samples")

BEST_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "best_hierarchical_model.pth")
BACKUP_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "backup_84_14_model.pth")

# 2. Cấu hình Dữ liệu (Dataset)
DATASET_CONFIG = {
    "mode": "c1",                  # "c1" (37 lớp chuẩn ICCV 2025) hoặc "full" (57 lớp)
    "train_ratio": 0.80,
    "val_ratio": 0.10,
    "test_ratio": 0.10,
    "seed": 42,
    "img_size": 224,               # Kích thước ảnh chuẩn 224x224
    "num_workers": 2,
    "batch_size": 32
}

# 3. Cấu hình Mô hình (Model Architecture)
MODEL_CONFIG = {
    "backbone": "resnet50",        # Backbone feature extractor: resnet50, convnext_base
    "feature_dim": 256,            # Số chiều biểu diễn không gian ẩn
    "texture_dim": 64,             # Số chiều vector vi kết cấu cục bộ (LBP + GLCM + Gabor)
    "gat_num_layers": 3,           # Số tầng GAT với residual connection
    "gat_heads": 2,                # Số attention heads
    "dropout": 0.10,
    "logit_scale_init": 2.65926    # Scale nhiệt độ khởi tạo cho Cosine Similarity (~14.3)
}

# 4. Cấu hình Hàm mất mát (Loss Configuration)
LOSS_CONFIG = {
    "default_type": "greedy",      # "greedy", "tree_dist", "focal"
    "tree_dist_alpha": 0.30,       # Hệ số phạt khoảng cách cây
    "focal_gamma": 2.0,            # Hệ số focusing cho Focal Loss
    "label_smoothing": 0.05,       # Làm mượt nhãn nhẹ
    "num_levels": 5                # Phase -> State -> Composition -> Form -> Material
}

# 5. Cấu hình Huấn luyện (Training)
TRAIN_CONFIG = {
    "epochs": 25,
    "lr_backbone": 1e-4,
    "lr_gat": 1e-3,
    "weight_decay": 1e-2,
    "eta_min": 1e-6,
    "mixed_precision": True,
    "use_scaler": True
}

# 6. Cấu hình Giao diện và Suy luận (Inference & Gradio)
APP_CONFIG = {
    "server_name": "127.0.0.1",
    "server_port": 7860,
    "share": False,
    "default_confidence_threshold": 0.45,
    "mc_dropout_passes": 10,
    "probing_scales": [64, 128, 256, 512, 1024],
    "dense_heatmap_default_grid": 8,
    "dense_heatmap_default_alpha": 0.45
}
