import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import torch

# 1. Danh sách các lớp bị loại bỏ trong cấu hình Matador-C1 (ICCV 2025 - Section S2.5)
# Loại bỏ 5 lớp vật liệu phẳng không có kết cấu đặc trưng từ hình ảnh cục bộ:
OMITTED_CLASSES = {
    "glass",
    "paint",
    "thermoplastic",
    "thermoset",
    "elastomer"
}

# 2. Bảng ánh xạ gộp các lớp tương đồng về diện mạo/kết cấu trong Matador-C1 (Section S2.5):
CONSOLIDATION_MAP = {
    # Gộp tất cả kim loại về lớp 'metal' (Generic metal)
    "aluminum": "metal",
    "brass": "metal",
    "bronze": "metal",
    "copper": "metal",
    "iron": "metal",
    "steel": "metal",

    # Gốm sứ: gộp porcelain, terracotta, stoneware về 'pottery'
    "porcelain": "pottery",
    "terracotta": "pottery",
    "stoneware": "pottery",

    # Đất: gộp dirt, soil về 'soil'
    "dirt": "soil",
    "soil": "soil",

    # Cây bụi & lá: gộp shrub, foliage về 'foliage'
    "shrub": "foliage",
    "foliage": "foliage",

    # Đá trầm tích: gộp sandstone, shale về 'shale'
    "sandstone": "shale",
    "shale": "shale",

    # Đá biến chất/kết tinh: gộp marble, quartz về 'marble'
    "marble": "marble",
    "quartz": "marble",

    # Xây dựng: gộp cement, concrete về 'concrete'
    "cement": "concrete",
    "concrete": "concrete",

    # Giấy & bìa: gộp cardboard, paper về 'paper'
    "cardboard": "paper",
    "paper": "paper",

    # Sợi bóng / Satin: gộp polyester, silk về 'satin'
    "polyester": "satin",
    "silk": "satin",

    # Sợi tự nhiên: gộp cotton, linen về 'natural_fiber'
    "cotton": "natural_fiber",
    "linen": "natural_fiber"
}

# 3. Cấu hình Matador-C1 (37 lớp lá) theo chuẩn bài báo ICCV 2025
MATADOR_C1_TREE = {
    "Solid": {
        "Abiotic": {
            "Metal": {
                "Metal": ["metal"]
            },
            "Rock": {
                "Solid Mass": ["granite", "limestone", "marble"],
                "Aggregate": ["asphalt", "concrete", "gravel", "sand", "shale"]
            },
            "Ceramic": {
                "Decorative": ["pottery"],
                "Structural": ["brick", "plaster"]
            }
        },
        "Biotic": {
            "Polymer": {
                "Textile": ["carpet", "natural_fiber", "satin", "wool"],
                "Synthetic": ["carbon_fiber", "foam", "nylon", "wax"]
            },
            "Natural": {
                "Vegetation": [
                    "flower", "foliage", "grass", "ivy", "moss",
                    "plant_litter", "straw", "tree_bark"
                ],
                "Terrain": ["soil"],
                "Wood": ["paper", "timber"]
            },
            "Derivative": {
                "Animal Hide": ["fur", "leather", "suede"],
                "Food": ["bread", "fruit", "vegetable"]
            }
        }
    }
}

# Cấu hình đầy đủ 57 lớp (Matador)
MATADOR_57_TREE = {
    "Solid": {
        "Abiotic": {
            "Metal": {
                "Ferrous": ["steel", "iron"],
                "Non-ferrous": ["aluminum", "brass", "bronze", "copper"]
            },
            "Rock": {
                "Solid Mass": ["granite", "limestone", "marble", "quartz", "shale"],
                "Aggregate": ["asphalt", "concrete", "gravel", "sand", "sandstone"]
            },
            "Ceramic": {
                "Decorative": ["porcelain", "terracotta", "stoneware"],
                "Structural": ["brick", "plaster", "cement"]
            },
            "Glass": {
                "Glass": ["glass"]
            }
        },
        "Biotic": {
            "Polymer": {
                "Textile": ["cotton", "linen", "silk", "wool", "carpet"],
                "Synthetic": [
                    "nylon", "polyester", "carbon_fiber", "elastomer", 
                    "foam", "thermoplastic", "thermoset", "paint", "wax"
                ]
            },
            "Natural": {
                "Vegetation": [
                    "flower", "foliage", "grass", "ivy", "moss", 
                    "shrub", "tree_bark", "plant_litter", "straw"
                ],
                "Terrain": ["dirt", "soil"],
                "Wood": ["timber", "paper", "cardboard"]
            },
            "Derivative": {
                "Animal Hide": ["fur", "leather", "suede"],
                "Food": ["bread", "fruit", "vegetable"]
            }
        }
    }
}

# Mặc định sử dụng cấu hình Matador-C1 (37 lớp) để loại bỏ nhiễu nhãn theo yêu cầu bài báo
taxonomy_tree = MATADOR_C1_TREE

# 4. Bảng tra cứu thuộc tính cơ học (density kg/m3, youngs_modulus GPa, tensile_strength MPa)
material_properties = {
    "metal": {"density": 7850, "youngs_modulus": 150.0, "tensile_strength": 300.0},
    "aluminum": {"density": 2700, "youngs_modulus": 70.0, "tensile_strength": 90.0},
    "brass": {"density": 8500, "youngs_modulus": 100.0, "tensile_strength": 350.0},
    "bronze": {"density": 8800, "youngs_modulus": 110.0, "tensile_strength": 400.0},
    "copper": {"density": 8960, "youngs_modulus": 110.0, "tensile_strength": 220.0},
    "iron": {"density": 7870, "youngs_modulus": 200.0, "tensile_strength": 350.0},
    "steel": {"density": 7850, "youngs_modulus": 200.0, "tensile_strength": 400.0},
    "granite": {"density": 2650, "youngs_modulus": 50.0, "tensile_strength": 10.0},
    "limestone": {"density": 2500, "youngs_modulus": 45.0, "tensile_strength": 8.0},
    "marble": {"density": 2700, "youngs_modulus": 60.0, "tensile_strength": 12.0},
    "quartz": {"density": 2650, "youngs_modulus": 75.0, "tensile_strength": 50.0},
    "asphalt": {"density": 2300, "youngs_modulus": 5.0, "tensile_strength": 2.0},
    "concrete": {"density": 2400, "youngs_modulus": 30.0, "tensile_strength": 4.0},
    "cement": {"density": 2000, "youngs_modulus": 20.0, "tensile_strength": 3.0},
    "gravel": {"density": 1800, "youngs_modulus": 0.1, "tensile_strength": 0.5},
    "sand": {"density": 1600, "youngs_modulus": 0.05, "tensile_strength": 0.2},
    "sandstone": {"density": 2200, "youngs_modulus": 20.0, "tensile_strength": 5.0},
    "shale": {"density": 2400, "youngs_modulus": 15.0, "tensile_strength": 6.0},
    "pottery": {"density": 2200, "youngs_modulus": 40.0, "tensile_strength": 30.0},
    "porcelain": {"density": 2400, "youngs_modulus": 70.0, "tensile_strength": 60.0},
    "terracotta": {"density": 2000, "youngs_modulus": 15.0, "tensile_strength": 15.0},
    "stoneware": {"density": 2300, "youngs_modulus": 40.0, "tensile_strength": 25.0},
    "brick": {"density": 1800, "youngs_modulus": 12.0, "tensile_strength": 5.0},
    "plaster": {"density": 1400, "youngs_modulus": 6.0, "tensile_strength": 2.5},
    "carpet": {"density": 1200, "youngs_modulus": 0.5, "tensile_strength": 15.0},
    "natural_fiber": {"density": 1500, "youngs_modulus": 7.0, "tensile_strength": 310.0},
    "cotton": {"density": 1500, "youngs_modulus": 6.0, "tensile_strength": 300.0},
    "linen": {"density": 1500, "youngs_modulus": 8.0, "tensile_strength": 320.0},
    "satin": {"density": 1340, "youngs_modulus": 7.0, "tensile_strength": 330.0},
    "polyester": {"density": 1380, "youngs_modulus": 4.0, "tensile_strength": 160.0},
    "silk": {"density": 1300, "youngs_modulus": 10.0, "tensile_strength": 500.0},
    "wool": {"density": 1300, "youngs_modulus": 3.0, "tensile_strength": 150.0},
    "carbon_fiber": {"density": 1600, "youngs_modulus": 150.0, "tensile_strength": 600.0},
    "foam": {"density": 50, "youngs_modulus": 0.005, "tensile_strength": 0.3},
    "nylon": {"density": 1150, "youngs_modulus": 3.0, "tensile_strength": 80.0},
    "wax": {"density": 900, "youngs_modulus": 0.2, "tensile_strength": 2.0},
    "flower": {"density": 700, "youngs_modulus": 0.02, "tensile_strength": 0.5},
    "foliage": {"density": 750, "youngs_modulus": 0.1, "tensile_strength": 2.0},
    "shrub": {"density": 700, "youngs_modulus": 0.5, "tensile_strength": 4.0},
    "grass": {"density": 900, "youngs_modulus": 0.05, "tensile_strength": 0.8},
    "ivy": {"density": 800, "youngs_modulus": 0.1, "tensile_strength": 1.5},
    "moss": {"density": 600, "youngs_modulus": 0.01, "tensile_strength": 0.1},
    "plant_litter": {"density": 400, "youngs_modulus": 0.02, "tensile_strength": 0.4},
    "straw": {"density": 200, "youngs_modulus": 0.1, "tensile_strength": 1.0},
    "tree_bark": {"density": 600, "youngs_modulus": 1.5, "tensile_strength": 5.0},
    "soil": {"density": 1500, "youngs_modulus": 0.01, "tensile_strength": 0.1},
    "dirt": {"density": 1400, "youngs_modulus": 0.015, "tensile_strength": 0.1},
    "paper": {"density": 750, "youngs_modulus": 3.8, "tensile_strength": 22.0},
    "cardboard": {"density": 700, "youngs_modulus": 4.0, "tensile_strength": 20.0},
    "timber": {"density": 650, "youngs_modulus": 11.0, "tensile_strength": 90.0},
    "fur": {"density": 800, "youngs_modulus": 0.2, "tensile_strength": 10.0},
    "leather": {"density": 900, "youngs_modulus": 0.05, "tensile_strength": 25.0},
    "suede": {"density": 850, "youngs_modulus": 0.04, "tensile_strength": 20.0},
    "bread": {"density": 300, "youngs_modulus": 0.001, "tensile_strength": 0.05},
    "fruit": {"density": 950, "youngs_modulus": 0.01, "tensile_strength": 0.2},
    "vegetable": {"density": 950, "youngs_modulus": 0.01, "tensile_strength": 0.2}
}

# 5. Phân loại nhị phân thuộc tính cơ tính theo chuẩn ICCV 2025
def get_binary_mechanical_labels(material_name):
    norm_name = str(material_name).strip().lower()
    norm_name = CONSOLIDATION_MAP.get(norm_name, norm_name)
    if norm_name not in material_properties:
        return [0, 0, 0]
    props = material_properties[norm_name]
    strong = 1 if props["tensile_strength"] >= 1.0 else 0
    rigid = 1 if props["youngs_modulus"] >= 12.0 else 0
    heavy = 1 if props["density"] >= 1600.0 else 0
    return [strong, rigid, heavy]

# 6. Trích xuất đường đi phân cấp từ nút gốc đến nút lá
def find_taxonomy_path(tree, target_leaf):
    target_clean = str(target_leaf).strip().lower()
    target_clean = CONSOLIDATION_MAP.get(target_clean, target_clean)
    def search(subtree, current_path):
        if isinstance(subtree, dict):
            for node, children in subtree.items():
                res = search(children, current_path + [node])
                if res:
                    return res
        elif isinstance(subtree, list):
            for leaf in subtree:
                if leaf.lower() == target_clean:
                    return current_path + [leaf]
        return None
    return search(tree, [])

# 7. Xây dựng đồ thị có hướng G=(V, E) kèm nút toàn cục h_g
def build_graph_structure(tree):
    nodes = []
    edges = []
    node_levels = {}

    def traverse(subtree, parent_name=None, level=0):
        if isinstance(subtree, dict):
            for node_name, child_tree in subtree.items():
                if node_name not in node_levels:
                    node_levels[node_name] = level
                    nodes.append(node_name)
                if parent_name:
                    edges.append((parent_name, node_name))
                traverse(child_tree, node_name, level + 1)
        elif isinstance(subtree, list):
            for leaf_node in subtree:
                if leaf_node not in node_levels:
                    node_levels[leaf_node] = level
                    nodes.append(leaf_node)
                if parent_name:
                    edges.append((parent_name, leaf_node))

    traverse(tree)

    # Sắp xếp nút theo level rồi theo alphabet
    nodes = sorted(list(set(nodes)), key=lambda x: (node_levels[x], x))
    node_to_idx = {node: i for i, node in enumerate(nodes)}

    # Cạnh có hướng trong cây phân cấp
    edge_indices = [[node_to_idx[src], node_to_idx[dst]] for src, dst in edges]

    # Thêm nút toàn cục (global node) h_g ở vị trí cuối cùng
    global_node_idx = len(nodes)
    nodes.append("GLOBAL_NODE")
    node_to_idx["GLOBAL_NODE"] = global_node_idx
    node_levels["GLOBAL_NODE"] = -1

    # Nối 2 chiều giữa nút toàn cục và toàn bộ các nút phân cấp
    for i in range(global_node_idx):
        edge_indices.append([i, global_node_idx])
        edge_indices.append([global_node_idx, i])

    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    return nodes, node_to_idx, node_levels, edge_index
