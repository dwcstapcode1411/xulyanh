"""
test_models.py - Kiểm thử các module mô hình: FeatureExtractor, HierarchicalGATModel, GreedyHierarchicalLoss.
"""
import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from taxonomy import MATADOR_C1_TREE, build_graph_structure
from models import FeatureExtractor, HierarchicalGATModel
from losses import GreedyHierarchicalLoss


class TestFeatureExtractor(unittest.TestCase):
    """Kiểm tra FeatureExtractor (ResNet50 backbone)."""

    @classmethod
    def setUpClass(cls):
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cls.fe = FeatureExtractor(model_name='resnet50', pretrained=False, feature_dim=256).to(cls.device)
        cls.fe.eval()

    def test_output_shape(self):
        """Đầu ra phải có shape (B, 256) cho feature_dim=256."""
        x = torch.randn(4, 3, 224, 224).to(self.device)
        with torch.no_grad():
            out = self.fe(x)
        self.assertEqual(out.shape, (4, 256), f"Expected (4, 256), got {out.shape}")

    def test_single_sample(self):
        """Phải hoạt động với batch_size=1 trong eval mode."""
        x = torch.randn(1, 3, 224, 224).to(self.device)
        with torch.no_grad():
            out = self.fe(x)
        self.assertEqual(out.shape, (1, 256))

    def test_output_finite(self):
        """Đầu ra phải không có NaN hoặc Inf."""
        x = torch.randn(2, 3, 224, 224).to(self.device)
        with torch.no_grad():
            out = self.fe(x)
        self.assertFalse(torch.isnan(out).any(), "Output contains NaN!")
        self.assertFalse(torch.isinf(out).any(), "Output contains Inf!")

    def test_different_input_sizes(self):
        """Phải hoạt động với các kích thước ảnh khác nhau (224, 384)."""
        for size in [224, 384]:
            x = torch.randn(2, 3, size, size).to(self.device)
            with torch.no_grad():
                out = self.fe(x)
            self.assertEqual(out.shape, (2, 256), f"Failed for input size {size}")


class TestHierarchicalGATModel(unittest.TestCase):
    """Kiểm tra HierarchicalGATModel."""

    @classmethod
    def setUpClass(cls):
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        nodes, node_to_idx, node_levels, edge_index = build_graph_structure(MATADOR_C1_TREE)
        cls.nodes = nodes
        cls.node_to_idx = node_to_idx
        cls.node_levels = node_levels
        cls.num_nodes = len(nodes)
        cls.num_hierarchy_nodes = cls.num_nodes - 1  # trừ GLOBAL_NODE
        cls.global_node_idx = node_to_idx["GLOBAL_NODE"]
        cls.edge_index = edge_index.to(cls.device)

        cls.gat = HierarchicalGATModel(
            num_nodes=cls.num_nodes,
            edge_index=cls.edge_index,
            global_node_idx=cls.global_node_idx,
            feature_dim=256,
            num_layers=3,
            dropout=0.1
        ).to(cls.device)
        cls.gat.eval()

    def test_output_shape(self):
        """Đầu ra logits phải có shape (B, num_hierarchy_nodes)."""
        phi_x = torch.randn(4, 256).to(self.device)
        with torch.no_grad():
            logits = self.gat(phi_x)
        self.assertEqual(logits.shape, (4, self.num_hierarchy_nodes),
                         f"Expected (4, {self.num_hierarchy_nodes}), got {logits.shape}")

    def test_single_sample_eval(self):
        """Phải hoạt động với batch_size=1 trong eval mode (không lỗi BatchNorm)."""
        phi_x = torch.randn(1, 256).to(self.device)
        with torch.no_grad():
            logits = self.gat(phi_x)
        self.assertEqual(logits.shape, (1, self.num_hierarchy_nodes))

    def test_output_finite(self):
        """Đầu ra phải không có NaN hoặc Inf."""
        phi_x = torch.randn(2, 256).to(self.device)
        with torch.no_grad():
            logits = self.gat(phi_x)
        self.assertFalse(torch.isnan(logits).any(), "Logits contain NaN!")
        self.assertFalse(torch.isinf(logits).any(), "Logits contain Inf!")

    def test_num_hierarchy_nodes_matches_tree(self):
        """Số nút phân cấp phải khớp với cấu trúc cây (không tính GLOBAL_NODE)."""
        leaf_count = len([n for n, l in self.node_levels.items() if l == 4 and n != "GLOBAL_NODE"])
        self.assertEqual(leaf_count, 37)
        # Tổng nodes bao gồm: 1 root + 2 state + 6 comp + 11 form + 37 leaves = 57
        # Thực tế số nút phụ thuộc vào cây cụ thể
        self.assertGreater(self.num_hierarchy_nodes, leaf_count)

    def test_gradient_flow(self):
        """Gradient phải truyền ngược thành công qua toàn bộ mạng."""
        self.gat.train()
        phi_x = torch.randn(2, 256, requires_grad=True).to(self.device)
        logits = self.gat(phi_x)
        loss = logits.sum()
        loss.backward()
        self.assertIsNotNone(phi_x.grad)
        self.assertFalse(torch.isnan(phi_x.grad).any(), "Gradients contain NaN!")
        self.gat.eval()


class TestGreedyHierarchicalLoss(unittest.TestCase):
    """Kiểm tra hàm mất mát GreedyHierarchicalLoss."""

    def setUp(self):
        self.criterion = GreedyHierarchicalLoss(num_levels=5, label_smoothing=0.05)
        nodes, node_to_idx, node_levels, _ = build_graph_structure(MATADOR_C1_TREE)
        self.num_hierarchy_nodes = len(nodes) - 1

        # Tạo level_nodes_map
        self.level_nodes_map = {}
        for node_name, lvl in node_levels.items():
            if lvl >= 0 and node_name != "GLOBAL_NODE":
                if lvl not in self.level_nodes_map:
                    self.level_nodes_map[lvl] = []
                self.level_nodes_map[lvl].append(node_to_idx[node_name])
        for lvl in self.level_nodes_map:
            self.level_nodes_map[lvl] = torch.tensor(sorted(self.level_nodes_map[lvl]), dtype=torch.long)

    def test_loss_finite(self):
        """Loss phải là giá trị hữu hạn (không NaN, không Inf)."""
        logits = torch.randn(4, self.num_hierarchy_nodes)
        multi_hot = torch.zeros(4, self.num_hierarchy_nodes)
        multi_hot[:, 0] = 1.0  # dummy
        level_indices = torch.zeros(4, 5, dtype=torch.long)

        targets = {"multi_hot": multi_hot, "level_indices": level_indices}
        loss = self.criterion(logits, targets, self.level_nodes_map)

        self.assertFalse(torch.isnan(loss).any(), "Loss is NaN!")
        self.assertFalse(torch.isinf(loss).any(), "Loss is Inf!")
        self.assertGreater(loss.item(), 0, "Loss should be positive!")

    def test_loss_decreases_with_perfect_prediction(self):
        """Loss khi dự đoán hoàn hảo phải nhỏ hơn khi dự đoán ngẫu nhiên."""
        # Random prediction
        logits_random = torch.randn(4, self.num_hierarchy_nodes)
        multi_hot = torch.zeros(4, self.num_hierarchy_nodes)
        multi_hot[:, 0] = 1.0
        level_indices = torch.zeros(4, 5, dtype=torch.long)
        targets = {"multi_hot": multi_hot, "level_indices": level_indices}

        loss_random = self.criterion(logits_random, targets, self.level_nodes_map)

        # Near-perfect prediction
        logits_good = torch.full((4, self.num_hierarchy_nodes), -10.0)
        logits_good[:, 0] = 10.0
        loss_good = self.criterion(logits_good, targets, self.level_nodes_map)

        self.assertLess(loss_good.item(), loss_random.item(),
                        "Perfect prediction should have lower loss than random!")

    def test_loss_backward(self):
        """Loss phải hỗ trợ backward pass."""
        logits = torch.randn(4, self.num_hierarchy_nodes, requires_grad=True)
        multi_hot = torch.zeros(4, self.num_hierarchy_nodes)
        multi_hot[:, 0] = 1.0
        level_indices = torch.zeros(4, 5, dtype=torch.long)
        targets = {"multi_hot": multi_hot, "level_indices": level_indices}

        loss = self.criterion(logits, targets, self.level_nodes_map)
        loss.backward()
        self.assertIsNotNone(logits.grad)


class TestEndToEndForward(unittest.TestCase):
    """Kiểm tra luồng forward pass hoàn chỉnh: Image -> FeatureExtractor -> GAT -> Logits."""

    @classmethod
    def setUpClass(cls):
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        nodes, node_to_idx, _, edge_index = build_graph_structure(MATADOR_C1_TREE)
        cls.num_nodes = len(nodes)
        cls.global_node_idx = node_to_idx["GLOBAL_NODE"]
        cls.num_hierarchy = cls.num_nodes - 1

        cls.fe = FeatureExtractor(model_name='resnet50', pretrained=False, feature_dim=256).to(cls.device)
        cls.gat = HierarchicalGATModel(
            num_nodes=cls.num_nodes,
            edge_index=edge_index.to(cls.device),
            global_node_idx=cls.global_node_idx,
            feature_dim=256, num_layers=3
        ).to(cls.device)
        cls.fe.eval()
        cls.gat.eval()

    def test_full_pipeline(self):
        """Toàn bộ pipeline Image -> phi(x) -> logits phải hoạt động."""
        x = torch.randn(2, 3, 224, 224).to(self.device)
        with torch.no_grad():
            phi_x = self.fe(x)
            logits = self.gat(phi_x)
        self.assertEqual(logits.shape, (2, self.num_hierarchy))
        self.assertFalse(torch.isnan(logits).any())


if __name__ == "__main__":
    unittest.main(verbosity=2)
