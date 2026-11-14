import sys
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import timm

class FeatureExtractor(nn.Module):
    """
    Trích xuất đặc trưng phi(x) từ ảnh bằng mạng xương sống pretrained từ timm.
    Hỗ trợ: resnet50, convnext_base, efficientnet_b4.
    """
    def __init__(self, model_name='resnet50', pretrained=True, feature_dim=256):
        super(FeatureExtractor, self).__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        in_features = self.backbone.num_features  # 2048 cho ResNet-50, 1024 cho ConvNeXt-Base
        self.fc = nn.Sequential(
            nn.Linear(in_features, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )

    def forward(self, x):
        feat = self.backbone(x)
        return self.fc(feat)


class TextureAwareFeatureExtractor(nn.Module):
    """
    Bộ trích xuất đặc trưng kết hợp CNN backbone + Texture Features (LBP/GLCM/Gabor).
    
    Pipeline:  
      ảnh RGB -> backbone ResNet50 -> phi(x) 256-d ─┐
      ảnh RGB -> TextureFeatures -> 64-d -> MLP ──── ├── concat -> MLP_fuse -> 256-d
    
    Đặc trưng kết cấu giúp phân biệt các lớp vật liệu tương đồng ở tầng lá
    (ví dụ: granite vs concrete, leather vs suede) mà CNN đơn thuần bỏ qua.
    """
    def __init__(self, model_name='resnet50', pretrained=True, feature_dim=256, texture_dim=64):
        super(TextureAwareFeatureExtractor, self).__init__()
        self.feature_dim = feature_dim
        self.texture_dim = texture_dim
        
        # CNN backbone
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        in_features = self.backbone.num_features
        self.fc_cnn = nn.Sequential(
            nn.Linear(in_features, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        
        # Texture feature projection (64-d -> feature_dim)
        self.fc_texture = nn.Sequential(
            nn.Linear(texture_dim, feature_dim // 2),
            nn.LayerNorm(feature_dim // 2),
            nn.GELU(),
            nn.Linear(feature_dim // 2, feature_dim),
            nn.Dropout(0.1)
        )
        
        # Fusion layer (concat 2 * feature_dim -> feature_dim)
        self.fuse = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Dropout(0.15)
        )
    
    def forward(self, x, texture_feats=None):
        """
        Args:
            x: Ảnh tensor (B, 3, H, W)
            texture_feats: Đặc trưng kết cấu đã tính trước (B, texture_dim) hoặc None
        Returns:
            phi_x: Đặc trưng tổng hợp (B, feature_dim)
        """
        cnn_feat = self.backbone(x)
        cnn_feat = self.fc_cnn(cnn_feat)  # (B, feature_dim)
        
        if texture_feats is not None:
            tex_feat = self.fc_texture(texture_feats)  # (B, feature_dim)
            combined = torch.cat([cnn_feat, tex_feat], dim=-1)  # (B, 2*feature_dim)
            return self.fuse(combined)  # (B, feature_dim)
        else:
            return cnn_feat

class HierarchicalGATLayer(nn.Module):
    """
    Một tầng GAT với edge attention, MLP psi_a và kết nối tắt (residual connection).
    """
    def __init__(self, feature_dim=256, heads=2, dropout=0.1):
        super(HierarchicalGATLayer, self).__init__()
        self.gat = GATConv(feature_dim, feature_dim // heads, heads=heads, concat=True, dropout=dropout)
        self.psi_a = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

    def forward(self, h, edge_index):
        h_gat = self.gat(h, edge_index)
        h_proc = self.psi_a(h_gat)
        # Residual Connection
        return h + h_proc

class HierarchicalGATModel(nn.Module):
    """
    Mô hình mạng nơ-ron đồ thị phân cấp nâng cấp khắc phục triệt để hiện tượng Over-squashing:
    1. Điều kiện hóa trực tiếp mọi nút v_i với đặc trưng ảnh phi(x) qua MLP_in tại đầu vào GAT.
    2. Đầu ra kết hợp điểm tương đồng đồ thị (Cosine Similarity x scale nhiệt độ), nhánh phân loại tuyến tính trực tiếp và nhánh nơ-ron đồ thị.
    """
    def __init__(self, num_nodes, edge_index, global_node_idx, feature_dim=256, num_layers=3, dropout=0.1):
        super(HierarchicalGATModel, self).__init__()
        self.num_nodes = num_nodes
        self.register_buffer('edge_index', edge_index)
        self.global_node_idx = global_node_idx
        self.feature_dim = feature_dim
        self.num_hierarchy_nodes = num_nodes - 1

        # Vector đặc trưng ban đầu h_i^0 cho các nút phân cấp (được nạp từ trung bình ảnh thật)
        self.node_embeddings = nn.Parameter(torch.randn(self.num_hierarchy_nodes, feature_dim) * 0.02)
        self.global_embedding = nn.Parameter(torch.zeros(1, feature_dim))

        # MLP_in: Điều kiện hóa trực tiếp đặc trưng ảnh phi(x) lên mọi nút phân cấp
        self.mlp_in = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, feature_dim)
        )

        # Xếp chồng D tầng Hierarchical GAT với residual connections
        self.layers = nn.ModuleList([
            HierarchicalGATLayer(feature_dim=feature_dim, heads=2, dropout=dropout)
            for _ in range(num_layers)
        ])

        # MLP psi_b xử lý đặc trưng sau GAT
        self.psi_b = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU()
        )

        # Hệ số nhiệt độ học được cho Cosine Similarity (khởi tạo ~ 2.659 tương đương scale=14.3)
        self.logit_scale = nn.Parameter(torch.ones([]) * 2.65926)

        # Nhánh phân loại tuyến tính trực tiếp từ phi(x) tới các nút phân cấp (đảm bảo gradient mạnh tới tầng lá)
        self.direct_classifier = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, self.num_hierarchy_nodes)
        )

        # Nhánh phân loại từ vector nút h_i^D sau GAT
        self.node_classifier = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim // 2, 1)
        )

    def forward(self, phi_x):
        """
        Args:
            phi_x: Đặc trưng ảnh từ FeatureExtractor (B, feature_dim)
        Returns:
            logits: Logits phân loại đa cấp cho tất cả các nút phân cấp (B, num_hierarchy_nodes)
        """
        batch_size = phi_x.size(0)

        # Ghép embeddings các nút phân cấp và nút toàn cục: (num_nodes, feature_dim)
        base_h = torch.cat([self.node_embeddings, self.global_embedding], dim=0)

        # 1. Điều kiện hóa trực tiếp mọi nút với phi(x)
        base_h_exp = base_h.unsqueeze(0).expand(batch_size, -1, -1) # (B, num_nodes, feature_dim)
        phi_x_exp = phi_x.unsqueeze(1).expand(-1, self.num_nodes, -1) # (B, num_nodes, feature_dim)
        concat_feat = torch.cat([base_h_exp, phi_x_exp], dim=-1) # (B, num_nodes, 2*feature_dim)
        h_init = self.mlp_in(concat_feat) + base_h_exp # (B, num_nodes, feature_dim)

        # 2. Truyền tin qua D tầng GAT kèm kết nối tắt (Residual Connection)
        h_list = []
        for b in range(batch_size):
            h_b = h_init[b]
            for layer in self.layers:
                h_b = layer(h_b, self.edge_index)
            h_b = self.psi_b(h_b)
            h_list.append(h_b)

        h_out = torch.stack(h_list, dim=0) # (B, num_nodes, feature_dim)
        h_tree = h_out[:, :self.num_hierarchy_nodes] # (B, num_hierarchy_nodes, feature_dim)

        # 3. Tính toán các thành phần logits:
        # a) Điểm tương đồng đồ thị (Cosine Similarity x scale nhiệt độ)
        h_norm = F.normalize(h_tree, p=2, dim=-1) # (B, num_hierarchy_nodes, feature_dim)
        phi_norm = F.normalize(phi_x, p=2, dim=-1).unsqueeze(1) # (B, 1, feature_dim)
        cos_sim = torch.sum(h_norm * phi_norm, dim=-1) # (B, num_hierarchy_nodes)
        temp_scale = torch.clamp(self.logit_scale.exp(), max=100.0)
        graph_sim_logits = cos_sim * temp_scale

        # b) Nhánh phân loại tuyến tính trực tiếp từ phi(x)
        direct_logits = self.direct_classifier(phi_x) # (B, num_hierarchy_nodes)

        # c) Nhánh phân loại nút từ h_i^D
        node_logits = self.node_classifier(h_tree).squeeze(-1) # (B, num_hierarchy_nodes)

        # Kết hợp đa nguồn dự đoán
        final_logits = graph_sim_logits + direct_logits + 0.5 * node_logits
        return final_logits

def initialize_node_features(feature_extractor, dataloader, nodes, node_to_idx, node_levels, device):
    """
    Khởi tạo vector đặc trưng ban đầu h_i^0 bằng trung bình đặc trưng phi(x_j) 
    của các ảnh thật thuộc tập T_i trong tập Train.
    """
    print("-> Đang khởi tạo vector đặc trưng h_i^0 từ trung bình ảnh thật tập Train...")
    feature_extractor.eval()

    class_features = {}
    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            phi = feature_extractor(images).cpu()
            leaf_names = targets["leaf_name"]
            for i, name in enumerate(leaf_names):
                clean_name = str(name).strip().lower()
                if clean_name not in class_features:
                    class_features[clean_name] = []
                class_features[clean_name].append(phi[i])

    leaf_means = {}
    for name, feats in class_features.items():
        stacked = torch.stack(feats, dim=0)
        leaf_means[name] = stacked.mean(dim=0)

    num_hierarchy_nodes = len(nodes) - 1
    feature_dim = next(iter(leaf_means.values())).shape[0]
    initial_embeddings = torch.zeros(num_hierarchy_nodes, feature_dim)

    for node_name, idx in node_to_idx.items():
        if node_name == "GLOBAL_NODE":
            continue
        clean_name = str(node_name).strip().lower()
        if clean_name in leaf_means:
            initial_embeddings[idx] = leaf_means[clean_name]
        else:
            all_feats = [feat for feats in class_features.values() for feat in feats]
            if all_feats:
                initial_embeddings[idx] = torch.stack(all_feats[:400], dim=0).mean(dim=0)
            else:
                initial_embeddings[idx] = torch.randn(feature_dim) * 0.02

    print(f"-> Hoàn tất khởi tạo {num_hierarchy_nodes} nút phân cấp với kích thước feature={feature_dim}!")
    return initial_embeddings
