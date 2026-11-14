"""
test_improvements.py - Kiểm thử các module nâng cấp và cải thiện mới:
  1. texture_features.py: LBP, GLCM, Gabor Filters, trích xuất vector 64 chiều.
  2. losses.py: TreeDistanceLoss (phạt khoảng cách cây), FocalHierarchicalLoss (tập trung mẫu khó).
  3. models.py: TextureAwareFeatureExtractor (kết hợp CNN + Texture Descriptors).
"""
import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn

from texture_features import (
    compute_lbp,
    compute_lbp_histogram,
    compute_glcm_features,
    compute_gabor_features,
    extract_texture_features
)
from taxonomy import MATADOR_C1_TREE, build_graph_structure
from losses import TreeDistanceLoss, FocalHierarchicalLoss
from models import TextureAwareFeatureExtractor


class TestTextureFeatures(unittest.TestCase):
    """Kiểm tra module trích xuất đặc trưng kết cấu cục bộ (Xử lý ảnh)."""

    def setUp(self):
        # Tạo ảnh xám thử nghiệm 64x64
        np.random.seed(42)
        self.gray = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
        self.rgb = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)

    def test_compute_lbp_shape_and_range(self):
        """LBP phải giữ nguyên kích thước ảnh (H, W) và giá trị trong [0, 255]."""
        lbp = compute_lbp(self.gray, radius=1, n_points=8)
        self.assertEqual(lbp.shape, self.gray.shape)
        self.assertTrue((lbp >= 0).all() and (lbp <= 255).all())

    def test_compute_lbp_histogram_normalized(self):
        """Histogram LBP phải có đúng số bin và chuẩn hóa L1 = 1.0."""
        n_bins = 32
        hist = compute_lbp_histogram(self.gray, radius=1, n_points=8, n_bins=n_bins)
        self.assertEqual(len(hist), n_bins)
        self.assertAlmostEqual(float(hist.sum()), 1.0, places=4)

    def test_compute_glcm_features(self):
        """Đặc trưng GLCM (Haralick) phải tính được đầy đủ và không chứa NaN."""
        feats = compute_glcm_features(self.gray, distances=[1, 3])
        # 2 khoảng cách x 4 góc x 4 đặc trưng = 32
        self.assertEqual(len(feats), 32)
        self.assertFalse(np.isnan(feats).any())
        self.assertFalse(np.isinf(feats).any())

    def test_compute_gabor_features(self):
        """Đặc trưng bộ lọc Gabor phải tính được đúng số chiều và hữu hạn."""
        feats = compute_gabor_features(self.gray, num_orientations=4, num_scales=3)
        # 4 hướng x 3 tỉ lệ x 2 (mean, std) = 24
        self.assertEqual(len(feats), 24)
        self.assertFalse(np.isnan(feats).any())
        self.assertFalse(np.isinf(feats).any())

    def test_extract_texture_features_target_dim(self):
        """Hàm trích xuất tổng hợp phải trả về đúng target_dim=64 chuẩn hóa L2."""
        feat = extract_texture_features(self.rgb, target_dim=64)
        self.assertEqual(len(feat), 64)
        self.assertEqual(feat.dtype, np.float32)
        norm = np.linalg.norm(feat)
        self.assertGreater(norm, 0.0)
        self.assertLessEqual(norm, 1.01)


class TestTreeDistanceLoss(unittest.TestCase):
    """Kiểm tra TreeDistanceLoss có trọng số phạt khoảng cách trên cây phân cấp."""

    @classmethod
    def setUpClass(cls):
        cls.nodes, cls.node_to_idx, cls.node_levels, _ = build_graph_structure(MATADOR_C1_TREE)
        cls.tree_loss = TreeDistanceLoss(
            nodes=cls.nodes,
            node_to_idx=cls.node_to_idx,
            node_levels=cls.node_levels,
            tree=MATADOR_C1_TREE,
            alpha=0.3
        )
        cls.num_hierarchy_nodes = len(cls.nodes) - 1

    def test_distance_matrix_properties(self):
        """Ma trận khoảng cách cây phải đối xứng, đường chéo = 0, giá trị >= 0."""
        dist = self.tree_loss.distance_matrix
        self.assertEqual(dist.dim(), 2)
        self.assertEqual(dist.size(0), dist.size(1))
        # Đường chéo chính phải bằng 0 (khoảng cách tới chính mình)
        diag = torch.diag(dist)
        self.assertTrue((diag == 0).all())
        # Đối xứng
        self.assertTrue(torch.allclose(dist, dist.t(), atol=1e-5))
        # Giá trị trong đoạn [0, 1]
        self.assertTrue((dist >= 0).all())
        self.assertTrue((dist <= 1.0).all())

    def test_tree_distance_loss_forward(self):
        """TreeDistanceLoss forward pass phải trả về scalar hữu hạn > 0."""
        batch_size = 4
        logits = torch.randn(batch_size, self.num_hierarchy_nodes, requires_grad=True)
        multi_hot = torch.zeros(batch_size, self.num_hierarchy_nodes)
        
        # Chọn 1 nhãn lá hợp lệ cho mỗi mẫu
        leaf_indices = self.tree_loss.leaf_indices
        level_indices = torch.zeros(batch_size, 5, dtype=torch.long)
        for b in range(batch_size):
            chosen_leaf = leaf_indices[b % len(leaf_indices)]
            multi_hot[b, chosen_leaf] = 1.0
            level_indices[b, 4] = chosen_leaf

        targets = {"multi_hot": multi_hot, "level_indices": level_indices}
        loss = self.tree_loss(logits, targets)

        self.assertFalse(torch.isnan(loss).any())
        self.assertFalse(torch.isinf(loss).any())
        self.assertGreater(loss.item(), 0.0)

        # Kiểm tra gradient backward
        loss.backward()
        self.assertIsNotNone(logits.grad)


class TestFocalHierarchicalLoss(unittest.TestCase):
    """Kiểm tra FocalHierarchicalLoss tập trung vào các mẫu khó phân loại."""

    @classmethod
    def setUpClass(cls):
        cls.nodes, cls.node_to_idx, cls.node_levels, _ = build_graph_structure(MATADOR_C1_TREE)
        cls.focal_loss = FocalHierarchicalLoss(node_levels=cls.node_levels, gamma=2.0)
        cls.num_hierarchy_nodes = len(cls.nodes) - 1

        cls.level_nodes_map = {}
        for node_name, lvl in cls.node_levels.items():
            if lvl >= 0 and node_name != "GLOBAL_NODE":
                if lvl not in cls.level_nodes_map:
                    cls.level_nodes_map[lvl] = []
                cls.level_nodes_map[lvl].append(cls.node_to_idx[node_name])
        for lvl in cls.level_nodes_map:
            cls.level_nodes_map[lvl] = torch.tensor(sorted(cls.level_nodes_map[lvl]), dtype=torch.long)

    def test_focal_loss_forward_and_backward(self):
        """Focal loss forward pass và backward pass phải thành công."""
        batch_size = 4
        logits = torch.randn(batch_size, self.num_hierarchy_nodes, requires_grad=True)
        multi_hot = torch.zeros(batch_size, self.num_hierarchy_nodes)
        multi_hot[:, 0] = 1.0
        level_indices = torch.zeros(batch_size, 5, dtype=torch.long)

        targets = {"multi_hot": multi_hot, "level_indices": level_indices}
        loss = self.focal_loss(logits, targets, self.level_nodes_map)

        self.assertFalse(torch.isnan(loss).any())
        self.assertGreater(loss.item(), 0.0)

        loss.backward()
        self.assertIsNotNone(logits.grad)


class TestTextureAwareFeatureExtractor(unittest.TestCase):
    """Kiểm tra TextureAwareFeatureExtractor (hòa trộn CNN + Texture features)."""

    def test_forward_without_texture(self):
        """Chạy mà không truyền texture_feats vẫn hoạt động bình thường như CNN thuần."""
        model = TextureAwareFeatureExtractor(model_name='resnet50', pretrained=False, feature_dim=256)
        model.eval()
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        self.assertEqual(out.shape, (2, 256))

    def test_forward_with_texture(self):
        """Chạy khi kết hợp vector đặc trưng kết cấu 64-d phải tạo ra vector 256-d."""
        model = TextureAwareFeatureExtractor(model_name='resnet50', pretrained=False, feature_dim=256, texture_dim=64)
        model.eval()
        x = torch.randn(2, 3, 224, 224)
        tex = torch.randn(2, 64)
        with torch.no_grad():
            out = model(x, texture_feats=tex)
        self.assertEqual(out.shape, (2, 256))
        self.assertFalse(torch.isnan(out).any())


if __name__ == "__main__":
    unittest.main(verbosity=2)
