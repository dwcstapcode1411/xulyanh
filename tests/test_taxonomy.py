"""
test_taxonomy.py - Kiểm thử tính đúng đắn của cấu trúc phân loại cây phân cấp Matador-C1.
Bao gồm: cấu trúc cây, đường đi 5 tầng, thuộc tính cơ học, đồ thị GAT.
"""
import sys
import os
import unittest

# Thêm thư mục cha vào PATH để import các module dự án
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from taxonomy import (
    MATADOR_C1_TREE, MATADOR_57_TREE, OMITTED_CLASSES, CONSOLIDATION_MAP,
    taxonomy_tree, material_properties, find_taxonomy_path,
    build_graph_structure, get_binary_mechanical_labels
)


class TestMatadorC1TreeStructure(unittest.TestCase):
    """Kiểm tra cấu trúc cây Matador-C1 (37 lớp lá)."""

    def setUp(self):
        self.tree = MATADOR_C1_TREE
        self.nodes, self.node_to_idx, self.node_levels, self.edge_index = build_graph_structure(self.tree)

    def test_exactly_37_leaf_classes(self):
        """Matador-C1 phải có đúng 37 lớp lá (Level 4)."""
        leaves = [n for n, lvl in self.node_levels.items() if lvl == 4 and n != "GLOBAL_NODE"]
        self.assertEqual(len(leaves), 37, f"Expected 37 leaf classes, got {len(leaves)}: {sorted(leaves)}")

    def test_all_leaves_have_5_level_path(self):
        """Mọi lớp lá phải có đường đi 5 tầng từ gốc: Phase -> State -> Composition -> Form -> Material."""
        leaves = [n for n, lvl in self.node_levels.items() if lvl == 4 and n != "GLOBAL_NODE"]
        for leaf in leaves:
            path = find_taxonomy_path(self.tree, leaf)
            self.assertIsNotNone(path, f"Leaf '{leaf}' has no path to root!")
            self.assertEqual(len(path), 5, f"Leaf '{leaf}' path has {len(path)} levels (expected 5): {path}")
            self.assertEqual(path[0], "Solid", f"Root of path for '{leaf}' is '{path[0]}', expected 'Solid'")

    def test_root_is_solid(self):
        """Gốc cây phải là 'Solid' (Level 0 - Phase)."""
        root_nodes = [n for n, lvl in self.node_levels.items() if lvl == 0 and n != "GLOBAL_NODE"]
        self.assertEqual(len(root_nodes), 1, f"Expected 1 root, got {len(root_nodes)}: {root_nodes}")
        self.assertEqual(root_nodes[0], "Solid")

    def test_level_1_state_nodes(self):
        """Level 1 (State) phải có 2 nút: Abiotic và Biotic."""
        state_nodes = sorted([n for n, lvl in self.node_levels.items() if lvl == 1 and n != "GLOBAL_NODE"])
        self.assertEqual(state_nodes, ["Abiotic", "Biotic"])

    def test_level_2_composition_nodes(self):
        """Level 2 (Composition) phải có các nút đúng."""
        comp_nodes = sorted([n for n, lvl in self.node_levels.items() if lvl == 2 and n != "GLOBAL_NODE"])
        expected = sorted(["Metal", "Rock", "Ceramic", "Polymer", "Natural", "Derivative"])
        self.assertEqual(comp_nodes, expected, f"Expected {expected}, got {comp_nodes}")

    def test_omitted_classes_not_in_c1(self):
        """Các lớp bị loại bỏ (glass, paint, etc.) không được xuất hiện trong Matador-C1."""
        all_nodes = set(self.nodes)
        for omitted in OMITTED_CLASSES:
            self.assertNotIn(omitted, all_nodes, f"Omitted class '{omitted}' should not be in C1 tree!")

    def test_consolidation_map_targets_in_c1(self):
        """Mọi nhãn đích trong CONSOLIDATION_MAP phải xuất hiện trong cây C1."""
        all_nodes = set(self.nodes)
        for src, dst in CONSOLIDATION_MAP.items():
            self.assertIn(dst, all_nodes, f"Consolidation target '{dst}' (from '{src}') not found in C1 tree!")


class TestGraphStructure(unittest.TestCase):
    """Kiểm tra cấu trúc đồ thị GAT (nodes, edges, global node)."""

    def setUp(self):
        self.nodes, self.node_to_idx, self.node_levels, self.edge_index = build_graph_structure(MATADOR_C1_TREE)

    def test_global_node_exists(self):
        """Nút toàn cục GLOBAL_NODE phải tồn tại ở vị trí cuối cùng."""
        self.assertIn("GLOBAL_NODE", self.node_to_idx)
        global_idx = self.node_to_idx["GLOBAL_NODE"]
        self.assertEqual(global_idx, len(self.nodes) - 1)

    def test_edge_index_shape(self):
        """edge_index phải có shape (2, num_edges) với kiểu torch.long."""
        self.assertEqual(self.edge_index.dim(), 2)
        self.assertEqual(self.edge_index.size(0), 2)
        self.assertEqual(self.edge_index.dtype, torch.long)

    def test_edge_index_valid_range(self):
        """Mọi chỉ số trong edge_index phải nằm trong [0, num_nodes)."""
        num_nodes = len(self.nodes)
        self.assertTrue((self.edge_index >= 0).all())
        self.assertTrue((self.edge_index < num_nodes).all())

    def test_global_node_bidirectional_edges(self):
        """GLOBAL_NODE phải có cạnh 2 chiều tới mọi nút phân cấp."""
        global_idx = self.node_to_idx["GLOBAL_NODE"]
        num_hierarchy = len(self.nodes) - 1  # trừ GLOBAL_NODE

        # Đếm cạnh từ global đến các nút
        from_global = (self.edge_index[0] == global_idx).sum().item()
        to_global = (self.edge_index[1] == global_idx).sum().item()

        self.assertEqual(from_global, num_hierarchy,
                         f"GLOBAL_NODE should have {num_hierarchy} outgoing edges, got {from_global}")
        self.assertEqual(to_global, num_hierarchy,
                         f"GLOBAL_NODE should have {num_hierarchy} incoming edges, got {to_global}")

    def test_no_self_loops_in_tree_edges(self):
        """Không có cạnh tự lặp (self-loop) trong đồ thị."""
        src = self.edge_index[0]
        dst = self.edge_index[1]
        self.assertFalse((src == dst).any(), "Graph should not have self-loops!")

    def test_node_to_idx_bijective(self):
        """node_to_idx phải là ánh xạ 1-1 (bijective)."""
        indices = list(self.node_to_idx.values())
        self.assertEqual(len(indices), len(set(indices)), "node_to_idx has duplicate indices!")
        self.assertEqual(len(self.node_to_idx), len(self.nodes), "Mismatch between node_to_idx size and nodes list!")


class TestMaterialProperties(unittest.TestCase):
    """Kiểm tra bảng thuộc tính cơ học và hàm phân loại nhị phân."""

    def test_all_c1_leaves_have_properties(self):
        """Mọi lớp lá trong Matador-C1 phải có thuộc tính cơ học."""
        _, _, node_levels, _ = build_graph_structure(MATADOR_C1_TREE)
        leaves = [n for n, lvl in node_levels.items() if lvl == 4 and n != "GLOBAL_NODE"]
        for leaf in leaves:
            self.assertIn(leaf, material_properties,
                          f"Leaf class '{leaf}' has no entry in material_properties!")

    def test_properties_have_valid_keys(self):
        """Mỗi entry trong material_properties phải có 3 key: density, youngs_modulus, tensile_strength."""
        for mat, props in material_properties.items():
            for key in ["density", "youngs_modulus", "tensile_strength"]:
                self.assertIn(key, props, f"Material '{mat}' missing key '{key}'")
                self.assertIsInstance(props[key], (int, float), f"Material '{mat}' key '{key}' is not numeric!")
                self.assertGreaterEqual(props[key], 0, f"Material '{mat}' key '{key}' is negative!")

    def test_binary_mechanical_labels_shape(self):
        """get_binary_mechanical_labels phải trả về list 3 phần tử (Strong, Rigid, Heavy)."""
        labels = get_binary_mechanical_labels("metal")
        self.assertEqual(len(labels), 3)
        for v in labels:
            self.assertIn(v, [0, 1])

    def test_binary_labels_consolidation(self):
        """Nhãn gộp (e.g., 'aluminum' -> 'metal') phải trả về giá trị hợp lệ."""
        for src in CONSOLIDATION_MAP:
            labels = get_binary_mechanical_labels(src)
            self.assertEqual(len(labels), 3, f"Binary labels for '{src}' has wrong length!")

    def test_unknown_material_returns_zeros(self):
        """Vật liệu không tồn tại phải trả về [0, 0, 0]."""
        labels = get_binary_mechanical_labels("nonexistent_material_xyz")
        self.assertEqual(labels, [0, 0, 0])


class TestFindTaxonomyPath(unittest.TestCase):
    """Kiểm tra hàm tìm đường đi phân cấp."""

    def test_valid_leaf_returns_path(self):
        """Tìm đường đi cho lớp lá hợp lệ."""
        path = find_taxonomy_path(MATADOR_C1_TREE, "metal")
        self.assertIsNotNone(path)
        self.assertEqual(len(path), 5)
        self.assertEqual(path[0], "Solid")
        self.assertEqual(path[-1], "metal")

    def test_consolidation_leaf(self):
        """Tìm đường đi cho nhãn được gộp (e.g., 'aluminum' -> 'metal')."""
        path = find_taxonomy_path(MATADOR_C1_TREE, "aluminum")
        self.assertIsNotNone(path)
        self.assertEqual(path[-1], "metal")

    def test_nonexistent_leaf_returns_none(self):
        """Nhãn không tồn tại phải trả về None."""
        path = find_taxonomy_path(MATADOR_C1_TREE, "nonexistent_material")
        self.assertIsNone(path)

    def test_case_insensitive(self):
        """Tìm kiếm không phân biệt hoa/thường."""
        path_lower = find_taxonomy_path(MATADOR_C1_TREE, "brick")
        path_upper = find_taxonomy_path(MATADOR_C1_TREE, "Brick")
        self.assertEqual(path_lower, path_upper)

    def test_omitted_class_returns_none(self):
        """Các lớp bị loại bỏ trong C1 phải trả về None."""
        for omitted in OMITTED_CLASSES:
            path = find_taxonomy_path(MATADOR_C1_TREE, omitted)
            self.assertIsNone(path, f"Omitted class '{omitted}' should return None in C1 tree!")


class TestMatador57Tree(unittest.TestCase):
    """Kiểm tra cấu hình đầy đủ 57 lớp."""

    def test_57_tree_has_more_leaves(self):
        """Cây 57 lớp phải có nhiều lá hơn cây C1."""
        _, _, levels_57, _ = build_graph_structure(MATADOR_57_TREE)
        _, _, levels_c1, _ = build_graph_structure(MATADOR_C1_TREE)
        leaves_57 = len([n for n, l in levels_57.items() if l == 4 and n != "GLOBAL_NODE"])
        leaves_c1 = len([n for n, l in levels_c1.items() if l == 4 and n != "GLOBAL_NODE"])
        self.assertGreater(leaves_57, leaves_c1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
