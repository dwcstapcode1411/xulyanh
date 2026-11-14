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


class GreedyHierarchicalLoss(nn.Module):
    """
    Hàm mất mát phân cấp tham lam theo bài báo ICCV 2025:
    Loss = max(BCE(l_hat, l), (1/D) * sum_{d=0}^{D-1} CE(l_hat^d, l^d))
    kết hợp Label Smoothing nhẹ (0.05) để tránh cực đoan hóa phân bố ở các tầng sâu.
    """
    def __init__(self, node_levels=None, num_levels=5, label_smoothing=0.05):
        super(GreedyHierarchicalLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction='mean')
        self.num_levels = num_levels
        self.node_levels = node_levels
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets, level_nodes_map=None):
        """
        Args:
            logits: Tensor (B, num_hierarchy_nodes)
            targets: Dict chứa 'multi_hot' và 'level_indices' (B, D)
            level_nodes_map: Dict mapping level d -> list of node IDs at level d
        """
        multi_hot = targets["multi_hot"]
        level_indices = targets["level_indices"] # (B, D)
        batch_size = logits.size(0)
        num_hierarchy_nodes = logits.size(1)

        if multi_hot.size(1) > num_hierarchy_nodes:
            multi_hot = multi_hot[:, :num_hierarchy_nodes]

        # 1. Binary Cross Entropy trên toàn bộ đường dẫn nhãn đa cấp
        bce_loss = self.bce(logits, multi_hot)

        # 2. Cross Entropy (CE) cho từng tầng phân cấp (Level 0 đến D-1)
        ce_loss_sum = torch.tensor(0.0, device=logits.device)
        valid_d_count = 0

        D = level_indices.size(1)
        for d in range(D):
            target_ids = level_indices[:, d] # (B,)
            mask = target_ids >= 0
            if mask.sum() == 0:
                continue

            if level_nodes_map is not None and d in level_nodes_map and len(level_nodes_map[d]) > 1:
                d_nodes = level_nodes_map[d].to(logits.device)
                sub_logits = logits[mask][:, d_nodes] # (B_valid, num_candidates_at_d)

                valid_targets = target_ids[mask]
                local_targets = torch.zeros_like(valid_targets)
                for local_idx, global_idx in enumerate(d_nodes):
                    local_targets[valid_targets == global_idx] = local_idx

                ce_d = F.cross_entropy(
                    sub_logits,
                    local_targets,
                    label_smoothing=self.label_smoothing
                )
                ce_loss_sum = ce_loss_sum + ce_d
                valid_d_count += 1
            else:
                valid_targets = target_ids[mask]
                ce_d = F.cross_entropy(
                    logits[mask],
                    valid_targets,
                    label_smoothing=self.label_smoothing
                )
                ce_loss_sum = ce_loss_sum + ce_d
                valid_d_count += 1

        avg_ce_loss = ce_loss_sum / max(valid_d_count, 1)

        # 3. Hàm mất mát tham lam Greedy: lấy giá trị lớn nhất
        total_loss = torch.max(bce_loss, avg_ce_loss)
        return total_loss


class TreeDistanceLoss(nn.Module):
    """
    Tree-Distance Weighted Loss (Cải thiện #1):
    
    Ý tưởng: Khi mô hình dự đoán sai, mức phạt tỷ lệ thuận với khoảng cách 
    trên cây phân cấp giữa nhãn dự đoán và nhãn thật. Nhầm lẫn giữa 
    các lớp gần nhau (ví dụ: granite vs limestone, cùng Solid Mass) bị phạt
    nhẹ hơn so với nhầm lẫn giữa các lớp xa nhau (ví dụ: granite vs wool).
    
    Khoảng cách cây d_T(a, b) = depth(a) + depth(b) - 2 * depth(LCA(a, b))
    trong đó LCA (Lowest Common Ancestor) là tổ tiên chung gần nhất.
    
    Loss = GreedyHierarchicalLoss + α * TreeDistancePenalty
    """
    def __init__(self, nodes, node_to_idx, node_levels, tree, 
                 alpha=0.3, num_levels=5, label_smoothing=0.05):
        super(TreeDistanceLoss, self).__init__()
        self.alpha = alpha
        self.num_levels = num_levels
        
        # Loss cơ sở
        self.base_loss = GreedyHierarchicalLoss(
            node_levels=node_levels, 
            num_levels=num_levels, 
            label_smoothing=label_smoothing
        )
        
        # Xây dựng ma trận khoảng cách cây cho các nút lá (Level 4)
        self.leaf_nodes = []
        for name, lvl in node_levels.items():
            if lvl == 4 and name != "GLOBAL_NODE" and name in node_to_idx:
                self.leaf_nodes.append((name, node_to_idx[name]))
        
        self.leaf_indices = [idx for _, idx in self.leaf_nodes]
        self.num_leaves = len(self.leaf_indices)
        
        # Tạo bảng tra cứu: leaf_global_idx -> leaf_local_idx (vị trí trong mảng lá)
        self.leaf_global_to_local = {}
        for local_i, (_, global_i) in enumerate(self.leaf_nodes):
            self.leaf_global_to_local[global_i] = local_i
        
        # Xây dựng đường đi phân cấp cho mỗi lá
        from taxonomy import find_taxonomy_path
        self.leaf_paths = {}
        for name, idx in self.leaf_nodes:
            path = find_taxonomy_path(tree, name)
            if path:
                self.leaf_paths[idx] = path
        
        # Tính trước ma trận khoảng cách cây (num_leaves x num_leaves)
        dist_matrix = torch.zeros(self.num_leaves, self.num_leaves)
        for i, (name_i, idx_i) in enumerate(self.leaf_nodes):
            for j, (name_j, idx_j) in enumerate(self.leaf_nodes):
                if i == j:
                    continue
                path_i = self.leaf_paths.get(idx_i, [])
                path_j = self.leaf_paths.get(idx_j, [])
                # Tìm LCA depth
                lca_depth = 0
                for k in range(min(len(path_i), len(path_j))):
                    if path_i[k] == path_j[k]:
                        lca_depth = k + 1
                    else:
                        break
                d_tree = len(path_i) + len(path_j) - 2 * lca_depth
                dist_matrix[i, j] = d_tree
        
        # Chuẩn hóa khoảng cách về [0, 1]
        max_dist = dist_matrix.max()
        if max_dist > 0:
            dist_matrix = dist_matrix / max_dist
        
        self.register_buffer('distance_matrix', dist_matrix)
    
    def forward(self, logits, targets, level_nodes_map=None):
        """
        Args:
            logits: (B, num_hierarchy_nodes)
            targets: Dict với 'multi_hot', 'level_indices'
            level_nodes_map: Dict mapping level -> node indices
        """
        # 1. Loss cơ sở (Greedy Hierarchical)
        base = self.base_loss(logits, targets, level_nodes_map)
        
        # 2. Tree-Distance Penalty cho tầng lá (Level 4)
        level_indices = targets["level_indices"]  # (B, 5)
        target_leaf_ids = level_indices[:, 4]  # (B,) - nhãn thật Level 4
        
        batch_size = logits.size(0)
        
        # Lấy logits chỉ cho các nút lá
        leaf_idx_tensor = torch.tensor(self.leaf_indices, dtype=torch.long, device=logits.device)
        leaf_logits = logits[:, leaf_idx_tensor]  # (B, num_leaves)
        leaf_probs = F.softmax(leaf_logits, dim=-1)  # (B, num_leaves)
        
        # Tính phạt khoảng cách cây có trọng số xác suất
        tree_penalty = torch.tensor(0.0, device=logits.device)
        valid_count = 0
        
        for b in range(batch_size):
            gt_global = target_leaf_ids[b].item()
            if gt_global < 0 or gt_global not in self.leaf_global_to_local:
                continue
            
            gt_local = self.leaf_global_to_local[gt_global]
            
            # Phạt = sum_j( P(j) * d_tree(j, gt) ) cho j ≠ gt
            distances_to_gt = self.distance_matrix[gt_local]  # (num_leaves,)
            weighted_dist = (leaf_probs[b] * distances_to_gt).sum()
            
            tree_penalty = tree_penalty + weighted_dist
            valid_count += 1
        
        if valid_count > 0:
            tree_penalty = tree_penalty / valid_count
        
        # 3. Tổ hợp
        total = base + self.alpha * tree_penalty
        return total


class FocalHierarchicalLoss(nn.Module):
    """
    Focal Loss kết hợp phân cấp (Cải thiện bổ sung):
    
    Tập trung vào các mẫu khó phân loại (hard examples) ở tầng lá bằng 
    cách giảm trọng số cho các mẫu dễ (đã phân loại đúng với độ tin cậy cao).
    
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    
    Args:
        gamma: Hệ số focusing (mặc định 2.0)
        alpha_levels: Trọng số cho từng tầng [Phase, State, Comp, Form, Material]
    """
    def __init__(self, node_levels=None, num_levels=5, gamma=2.0, 
                 alpha_levels=None, label_smoothing=0.05):
        super(FocalHierarchicalLoss, self).__init__()
        self.gamma = gamma
        self.num_levels = num_levels
        self.node_levels = node_levels
        self.label_smoothing = label_smoothing
        
        # Trọng số tầng: tầng lá (Material) quan trọng nhất
        if alpha_levels is None:
            alpha_levels = [0.5, 0.8, 1.0, 1.2, 1.5]
        self.alpha_levels = alpha_levels
    
    def forward(self, logits, targets, level_nodes_map=None):
        multi_hot = targets["multi_hot"]
        level_indices = targets["level_indices"]
        num_hierarchy_nodes = logits.size(1)
        
        if multi_hot.size(1) > num_hierarchy_nodes:
            multi_hot = multi_hot[:, :num_hierarchy_nodes]
        
        # 1. BCE Loss
        bce_loss = F.binary_cross_entropy_with_logits(logits, multi_hot, reduction='mean')
        
        # 2. Focal CE cho từng tầng
        focal_loss_sum = torch.tensor(0.0, device=logits.device)
        valid_d_count = 0
        
        D = level_indices.size(1)
        for d in range(D):
            target_ids = level_indices[:, d]
            mask = target_ids >= 0
            if mask.sum() == 0:
                continue
            
            alpha_d = self.alpha_levels[d] if d < len(self.alpha_levels) else 1.0
            
            if level_nodes_map is not None and d in level_nodes_map and len(level_nodes_map[d]) > 1:
                d_nodes = level_nodes_map[d].to(logits.device)
                sub_logits = logits[mask][:, d_nodes]
                
                valid_targets = target_ids[mask]
                local_targets = torch.zeros_like(valid_targets)
                for local_idx, global_idx in enumerate(d_nodes):
                    local_targets[valid_targets == global_idx] = local_idx
                
                # Focal modulation
                ce = F.cross_entropy(sub_logits, local_targets, reduction='none',
                                     label_smoothing=self.label_smoothing)
                probs = F.softmax(sub_logits, dim=-1)
                p_t = probs.gather(1, local_targets.unsqueeze(1)).squeeze(1)
                focal_weight = (1.0 - p_t) ** self.gamma
                
                focal_ce_d = (alpha_d * focal_weight * ce).mean()
                focal_loss_sum = focal_loss_sum + focal_ce_d
                valid_d_count += 1
        
        avg_focal = focal_loss_sum / max(valid_d_count, 1)
        
        total = torch.max(bce_loss, avg_focal)
        return total
