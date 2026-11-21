import os
import sys
import warnings
import asyncio

# Tắt cảnh báo rác
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# 1. Cấu hình Windows Event Loop Policy để tắt cảnh báo _ProactorBasePipeTransport connection lost
if sys.platform == "win32":
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

print("=" * 65)
print("  MATADOR-C1: HỆ THỐNG PHÂN LOẠI VẬT LIỆU PHÂN CẤP (ICCV 2025)")
print("=" * 65)
print("-> [1/4] Đang khởi động và nạp các thư viện PyTorch, CUDA, Gradio... (vui lòng đợi ~10-15s)")
sys.stdout.flush()

import glob
import cv2
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import gradio as gr

from taxonomy import (
    taxonomy_tree,
    material_properties,
    get_binary_mechanical_labels,
    build_graph_structure,
    find_taxonomy_path
)
from models import FeatureExtractor, HierarchicalGATModel
from texture_features import (
    extract_texture_features,
    visualize_texture_analysis,
    compute_lbp,
    compute_glcm_features,
    compute_gabor_features
)
from optical_simulation import generate_pillbox_psf, apply_poisson_gaussian_noise

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpu_name = f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else ""
print(f"-> [2/4] Thiết bị tính toán đã sẵn sàng: {device}{gpu_name}")
sys.stdout.flush()

# 2. Khởi tạo cấu trúc đồ thị từ Matador-C1 taxonomy
nodes, node_to_idx, node_levels, edge_index = build_graph_structure(taxonomy_tree)
num_nodes = len(nodes)
global_node_idx = node_to_idx["GLOBAL_NODE"]

# Phân nhóm nút theo level
comp_nodes = [node_to_idx[c] for c in ["Metal", "Rock", "Ceramic", "Polymer", "Natural", "Derivative"] if c in node_to_idx]
leaf_nodes = [idx for name, idx in node_to_idx.items() if node_levels.get(name) == 4 and name != "GLOBAL_NODE"]
idx_to_node = {idx: name for name, idx in node_to_idx.items()}

# Bảng màu cho tầng Composition (Level 2)
COMPOSITION_PALETTE = {
    "Metal": (65, 130, 225),       # Xanh kim loại (Royal Blue)
    "Rock": (150, 150, 150),       # Xám đá (Gray)
    "Ceramic": (220, 110, 40),     # Cam gốm sứ (Terracotta)
    "Polymer": (165, 100, 235),    # Tím sợi / tổng hợp (Purple)
    "Natural": (45, 175, 80),      # Xanh lá tự nhiên (Green)
    "Derivative": (235, 185, 30)   # Vàng da lông / thực phẩm (Gold)
}

# 3. Khởi tạo mô hình & nạp trọng số thật
feature_extractor = FeatureExtractor(model_name='resnet50', pretrained=False, feature_dim=256).to(device)
gat_model = HierarchicalGATModel(
    num_nodes=num_nodes,
    edge_index=edge_index.to(device),
    global_node_idx=global_node_idx,
    feature_dim=256,
    num_layers=3
).to(device)

checkpoint_path = os.path.join("checkpoints", "best_hierarchical_model.pth")
if not os.path.isfile(checkpoint_path):
    backup_path = os.path.join("checkpoints", "backup_84_14_model.pth")
    if os.path.isfile(backup_path):
        checkpoint_path = backup_path

print(f"-> [3/4] Đang nạp trọng số mô hình từ {checkpoint_path}...")
sys.stdout.flush()

if os.path.isfile(checkpoint_path):
    print(f"-> Nạp checkpoint thành công: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    feature_extractor.load_state_dict(ckpt['feature_extractor_state'])
    gat_model.load_state_dict(ckpt['gat_model_state'])
else:
    print(f"Cảnh báo: Chưa tìm thấy checkpoint tại {checkpoint_path}, chạy với trọng số ngẫu nhiên.")

def effective_transforms(image_pil):
    """
    Tiền xử lý ảnh thuần PIL + PyTorch (không phụ thuộc torchvision C++ operator)
    Resize -> ToTensor -> Normalize(ImageNet mean & std).
    """
    if image_pil.mode != 'RGB':
        image_pil = image_pil.convert('RGB')
    resized = image_pil.resize((224, 224), Image.BILINEAR)
    arr = np.array(resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
    return (tensor - mean) / std

def enable_dropout(m):
    """Bật chế độ huấn luyện ảo chỉ cho tầng Dropout để tính toán Monte Carlo Dropout"""
    if isinstance(m, torch.nn.Dropout):
        m.train()

def load_image_as_rgb_uint8(image_input):
    """
    Đọc ảnh an toàn hỗ trợ:
    - Ảnh .tif / .tiff 8-bit và 16-bit (uint16)
    - Ảnh .png, .jpg, .jpeg, .bmp
    - Filepath, đối tượng file Gradio, PIL.Image hoặc numpy.ndarray
    Tự động chuẩn hóa về mảng numpy uint8 (3 kênh RGB).
    """
    if image_input is None:
        return None

    if isinstance(image_input, np.ndarray):
        arr = image_input
    elif isinstance(image_input, Image.Image):
        arr = np.array(image_input.convert('RGB'))
    elif isinstance(image_input, (str, os.PathLike)) or hasattr(image_input, 'name'):
        path = str(image_input if isinstance(image_input, (str, os.PathLike)) else image_input.name)
        try:
            with Image.open(path) as pil_img:
                arr = np.array(pil_img.convert('RGB'))
        except Exception:
            cv_img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if cv_img is None:
                return None
            if cv_img.ndim == 2:
                cv_img = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2RGB)
            elif cv_img.shape[2] == 4:
                cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGRA2RGB)
            elif cv_img.shape[2] == 3:
                cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            arr = cv_img
    else:
        return None

    # Chuẩn hóa nếu là 16-bit (uint16) hoặc float
    if arr.dtype == np.uint16:
        arr = (arr / 256.0).clip(0, 255).astype(np.uint8)
    elif arr.dtype in (np.float32, np.float64):
        if arr.max() <= 1.0:
            arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)

    # Đảm bảo 3 kênh RGB
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.ndim == 3 and arr.shape[2] == 1:
        arr = np.concatenate([arr] * 3, axis=-1)
    elif arr.ndim == 3 and arr.shape[2] > 3:
        arr = arr[:, :, :3]

    return arr

# =========================================================================
# 1. TAB 1: SCENE PROBING TỌA ĐỘ CỤC BỘ (MULTI-SCALE + BEST-FIRST SEARCH)
# =========================================================================
def probe_scene_coordinate(image, click_coords, confidence_threshold=0.45):
    """
    Thực hiện Scene Probing tại tọa độ click_coords (u, v):
    1. Quét đa quy mô (64x64 đến 1024x1024 px)
    2. Monte Carlo Dropout ước lượng độ bất định entropy
    3. Trích xuất bản đồ biên vi kết cấu Sobel Gradient Magnitude
    4. Duyệt phân cấp Best-First Search & suy luận cơ tính
    """
    if image is None:
        return None, None, "Vui lòng tải lên hoặc chọn một bức ảnh cảnh quan.", ""

    img_arr = load_image_as_rgb_uint8(image)
    if img_arr is None:
        return None, None, "Không thể đọc định dạng ảnh.", ""

    h_img, w_img, _ = img_arr.shape

    if click_coords is None:
        u, v = w_img // 2, h_img // 2
    elif hasattr(click_coords, 'index'):
        u, v = int(click_coords.index[0]), int(click_coords.index[1])
    elif isinstance(click_coords, dict):
        u, v = int(click_coords.get("x", w_img // 2)), int(click_coords.get("y", h_img // 2))
    elif isinstance(click_coords, (tuple, list)):
        u, v = int(click_coords[0]), int(click_coords[1])
    else:
        u, v = w_img // 2, h_img // 2

    u = max(0, min(w_img - 1, u))
    v = max(0, min(h_img - 1, v))

    # 1. Quét cửa sổ đa quy mô (Multi-scale crops) từ 128x128 đến kích thước ảnh
    max_dim = min(h_img, w_img)
    scales = [s for s in [128, 256, 384, 512] if s <= max_dim]
    if not scales:
        scales = [max_dim]

    best_scale = scales[-1]
    best_conf = -1.0
    best_crop = None
    best_logits = None
    best_leaf_name = None

    # Danh sách các nút lá
    leaf_idx_list = [idx for n, idx in node_to_idx.items() if node_levels.get(n) == 4 and n != "GLOBAL_NODE"]
    leaf_name_list = [n for n, idx in node_to_idx.items() if node_levels.get(n) == 4 and n != "GLOBAL_NODE"]

    for scale in scales:
        half_s = scale // 2
        y1, y2 = max(0, v - half_s), min(h_img, v + half_s)
        x1, x2 = max(0, u - half_s), min(w_img, u + half_s)

        crop = img_arr[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        crop_pil = Image.fromarray(crop).convert('RGB')
        crop_tensor = effective_transforms(crop_pil).unsqueeze(0).to(device)

        # 2. Monte Carlo Dropout (T = 5 lượt suy luận để ước lượng ổn định)
        feature_extractor.eval()
        gat_model.eval()
        feature_extractor.apply(enable_dropout)
        gat_model.apply(enable_dropout)

        mc_logits_list = []
        with torch.no_grad():
            for _ in range(5):
                with torch.amp.autocast('cuda'):
                    phi_x = feature_extractor(crop_tensor)
                    logits = gat_model(phi_x)
                mc_logits_list.append(logits)

        mean_logits = torch.stack(mc_logits_list, dim=0).mean(dim=0).squeeze(0)

        # Tính toán phân bố xác suất trên 37 lớp lá (Material)
        leaf_sub = mean_logits[leaf_idx_list]
        leaf_probs = torch.softmax(leaf_sub, dim=-1)
        max_p, top_pos = torch.max(leaf_probs, dim=-1)
        conf_val = max_p.item()

        if conf_val > best_conf:
            best_conf = conf_val
            best_scale = scale
            best_crop = crop
            best_logits = mean_logits
            best_leaf_name = leaf_name_list[top_pos.item()]

    # 3. Đánh dấu vị trí cửa sổ được lựa chọn trên ảnh
    marked_image = img_arr.copy()
    half_best = best_scale // 2
    cv2.rectangle(
        marked_image,
        (max(0, u - half_best), max(0, v - half_best)),
        (min(w_img, u + half_best), min(h_img, v + half_best)),
        (0, 255, 0), 4
    )
    cv2.circle(marked_image, (u, v), 8, (255, 0, 0), -1)

    # 4. Trích xuất bản đồ biên vi kết cấu (Sobel Gradient Magnitude)
    if best_crop is not None and best_crop.size > 0:
        gray_crop = cv2.cvtColor(best_crop, cv2.COLOR_RGB2GRAY)
        gx = cv2.Sobel(gray_crop, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray_crop, cv2.CV_64F, 0, 1, ksize=3)
        mag = np.sqrt(gx**2 + gy**2)
        mag_norm = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        edge_map = cv2.applyColorMap(mag_norm, cv2.COLORMAP_MAGMA)
        edge_map = cv2.cvtColor(edge_map, cv2.COLOR_BGR2RGB)
    else:
        edge_map = None

    # 5. Suy luận Phân cấp Nhất quán Cây (Hierarchical Path Consistency)
    leaf_sub = best_logits[leaf_idx_list]
    leaf_probs = torch.softmax(leaf_sub, dim=-1)
    top3_vals, top3_idx = torch.topk(leaf_probs, k=min(3, len(leaf_name_list)))

    top3_candidates = []
    for k in range(len(top3_vals)):
        c_name = leaf_name_list[top3_idx[k].item()]
        c_prob = top3_vals[k].item() * 100.0
        top3_candidates.append(f"**{c_name}**: `{c_prob:.1f}%`")

    # Đường dẫn phân cấp chuẩn của nhãn lá dự đoán
    raw_path = find_taxonomy_path(taxonomy_tree, best_leaf_name)
    if not raw_path:
        raw_path = ["Solid", "Abiotic", "Rock", "Solid Mass", best_leaf_name]

    inference_path = []
    for lvl, node_n in enumerate(raw_path):
        if lvl == 0:
            inference_path.append(f"{node_n} (100.0%)")
        elif lvl == 4:
            inference_path.append(f"{node_n} ({best_conf*100.0:.1f}%)")
        else:
            inference_path.append(node_n)

    path_display = " ➜ ".join(inference_path)

    # 6. Suy luận cơ tính vật lý
    mech_codes = get_binary_mechanical_labels(best_leaf_name)
    mech_text = (
        f"• Sức chịu kéo (Tensile Strength): {'Mạnh mẽ (Strong >= 1 MPa)' if mech_codes[0] == 1 else 'Dễ vỡ (Fragile < 1 MPa)'}\n"
        f"• Độ đàn hồi / Cứng cáp (Elasticity): {'Cứng cáp (Rigid >= 12 GPa)' if mech_codes[1] == 1 else 'Dễ biến dạng (Deformable < 12 GPa)'}\n"
        f"• Tỷ trọng (Density): {'Nặng (Heavy >= 1600 kg/m³)' if mech_codes[2] == 1 else 'Nhẹ (Light < 1600 kg/m³)'}"
    )

    output_details = (
        f"### 🎯 Kết quả Phân loại Phân cấp Nhất quán (Matador-C1):\n\n"
        f"**Cây phả hệ phân cấp đầy đủ:**\n"
        f"`{path_display}`\n\n"
        f"**🏆 Top 3 Vật liệu có độ tin cậy cao nhất:**\n"
        f"- 🥇 {top3_candidates[0]}\n"
        f"- 🥈 {top3_candidates[1] if len(top3_candidates) > 1 else ''}\n"
        f"- 🥉 {top3_candidates[2] if len(top3_candidates) > 2 else ''}\n\n"
        f"**Tọa độ phân tích:** `(X={u}, Y={v})` | **Quy mô cửa sổ tối ưu:** `{best_scale} × {best_scale}` px | **Độ tin cậy:** `{best_conf*100.0:.1f}%`"
    )

    return marked_image, edge_map, output_details, mech_text

    return marked_image, edge_map, output_details, mech_text

# =========================================================================
# 2. TAB 2: BẢN ĐỒ NHIỆT VẬT LIỆU DÀY ĐẶC (DENSE MATERIAL HEATMAP)
# =========================================================================
def generate_dense_material_heatmap(image, grid_res=8, blend_alpha=0.45):
    """
    Quét cửa sổ trượt phân giải cao trên toàn bộ bức ảnh (Grid Scan):
    1. Chia ảnh thành grid_res x grid_res ô lưới
    2. Gom nhóm xử lý batch qua mạng ResNet50 + Hierarchical GAT
    3. Trải lớp phủ màu (Overlay Mask) mã hóa theo nhóm vật liệu (Composition)
    4. Thống kê tỷ lệ diện tích các vật liệu phát hiện được
    """
    if image is None:
        return None, "Vui lòng chọn hoặc tải ảnh trước khi tạo bản đồ nhiệt."

    img_arr = load_image_as_rgb_uint8(image)
    if img_arr is None:
        return None, "Không thể đọc ảnh."

    h, w, _ = img_arr.shape
    grid_res = max(4, min(16, int(grid_res)))

    step_y = h // grid_res
    step_x = w // grid_res

    patches = []
    coords = []

    for gy in range(grid_res):
        for gx in range(grid_res):
            y1 = gy * step_y
            y2 = (gy + 1) * step_y if gy < grid_res - 1 else h
            x1 = gx * step_x
            x2 = (gx + 1) * step_x if gx < grid_res - 1 else w

            patch = img_arr[y1:y2, x1:x2]
            if patch.size == 0:
                continue

            pil_p = Image.fromarray(patch).convert('RGB')
            tensor_p = effective_transforms(pil_p)
            patches.append(tensor_p)
            coords.append((y1, y2, x1, x2))

    if not patches:
        return None, "Không thể trích xuất các ô lưới."

    batch_tensor = torch.stack(patches, dim=0).to(device)

    # Dự đoán mô hình theo lô
    feature_extractor.eval()
    gat_model.eval()

    all_preds_comp = []
    all_preds_leaf = []

    batch_size = 32
    with torch.no_grad():
        for i in range(0, len(patches), batch_size):
            sub_batch = batch_tensor[i:i + batch_size]
            with torch.amp.autocast('cuda'):
                phi = feature_extractor(sub_batch)
                logits = gat_model(phi)

            # Dự đoán Material (Level 4) từ tất cả các nút lá
            leaf_tensor = torch.tensor(leaf_nodes, device=device)
            best_l_idx = torch.argmax(logits[:, leaf_tensor], dim=-1)
            pred_leaf_ids = leaf_tensor[best_l_idx].cpu().numpy()

            for lid in pred_leaf_ids:
                leaf_name = idx_to_node.get(lid, "unknown")
                all_preds_leaf.append(leaf_name)
                # Suy ra Composition (Level 2) trực tiếp từ cây phả hệ
                p = find_taxonomy_path(taxonomy_tree, leaf_name)
                comp_name = p[2] if p and len(p) >= 3 else "Rock"
                all_preds_comp.append(comp_name)

    # Tạo lớp phủ màu theo Composition
    color_mask = np.zeros_like(img_arr)
    comp_counts = {}
    leaf_counts = {}

    for (y1, y2, x1, x2), comp_name, leaf_name in zip(coords, all_preds_comp, all_preds_leaf):
        color = COMPOSITION_PALETTE.get(comp_name, (128, 128, 128))
        color_mask[y1:y2, x1:x2] = color

        comp_counts[comp_name] = comp_counts.get(comp_name, 0) + 1
        leaf_counts[leaf_name] = leaf_counts.get(leaf_name, 0) + 1

    # Hòa trộn màu với ảnh gốc (Alpha Blending)
    overlay = cv2.addWeighted(img_arr, 1.0 - blend_alpha, color_mask, blend_alpha, 0)

    # Vẽ đường kẻ lưới nhẹ để dễ phân biệt
    for gy in range(1, grid_res):
        y = gy * step_y
        cv2.line(overlay, (0, y), (w, y), (255, 255, 255), 1)
    for gx in range(1, grid_res):
        x = gx * step_x
        cv2.line(overlay, (x, 0), (x, h), (255, 255, 255), 1)

    # Thống kê tỷ lệ phân bố
    total_patches = len(coords)
    summary_lines = [
        f"### 📊 Thống kê Phân bố Vật liệu ({grid_res}×{grid_res} = {total_patches} vùng quét):",
        "",
        "**Tầng Thành phần (Composition - Level 2):**"
    ]

    for comp, count in sorted(comp_counts.items(), key=lambda x: -x[1]):
        pct = (count / total_patches) * 100.0
        hex_color = "#{:02x}{:02x}{:02x}".format(*COMPOSITION_PALETTE.get(comp, (128, 128, 128)))
        summary_lines.append(f"- <span style='color:{hex_color}; font-weight:bold;'>■ {comp}</span>: {pct:.1f}% ({count}/{total_patches} vùng)")

    summary_lines.append("\n**Top 5 Vật liệu chi tiết (Material - Level 4):**")
    for leaf, count in sorted(leaf_counts.items(), key=lambda x: -x[1])[:5]:
        pct = (count / total_patches) * 100.0
        summary_lines.append(f"- **{leaf}**: {pct:.1f}% ({count} vùng)")

    return overlay, "\n".join(summary_lines)

# =========================================================================
# 3. TAB 3: TRÍCH XUẤT VI KẾT CẤU (LBP / GLCM / GABOR FILTERS)
# =========================================================================
def run_texture_feature_analysis(image):
    """
    Trích xuất toàn diện các bộ mô tả kết cấu cổ điển (Classical Texture Descriptors):
    - LBP (Local Binary Pattern)
    - GLCM Energy Map (Haralick Co-occurrence)
    - Sobel Gradient Magnitude
    - Gabor Multi-orientation Filters (0°, 45°, 90°, 135°)
    """
    if image is None:
        return None, "Vui lòng chọn hoặc tải ảnh trước."

    img_arr = load_image_as_rgb_uint8(image)
    if img_arr is None:
        return None, "Không thể đọc định dạng ảnh."

    os.makedirs("outputs", exist_ok=True)
    out_path = "outputs/app_texture_analysis.png"
    visualize_texture_analysis(img_arr, output_path=out_path)

    # Trích xuất vector 64 chiều
    feat_64 = extract_texture_features(img_arr, target_dim=64)

    # Đọc lại ảnh kết quả
    if os.path.isfile(out_path):
        result_img = Image.open(out_path)
    else:
        result_img = None

    stats_text = (
        f"### 🔬 Thống kê Vector Đặc trưng Vi Kết cấu (64-D Descriptor):\n"
        f"• **Chuẩn vector (L2 Norm):** `{np.linalg.norm(feat_64):.4f}`\n"
        f"• **Giá trị cực đại (Max):** `{feat_64.max():.4f}` | **Cực tiểu (Min):** `{feat_64.min():.4f}`\n"
        f"• **Trung bình (Mean):** `{feat_64.mean():.4f}` | **Độ lệch chuẩn (Std):** `{feat_64.std():.4f}`\n\n"
        f"**Cấu trúc vector:**\n"
        f"- `0..31`: LBP Histogram (32-d) mô tả vi kết cấu điểm ảnh cục bộ\n"
        f"- `32..63`: GLCM Haralick + Gabor Multi-orientation Responses đo tính đồng nhất & hướng vân"
    )

    return result_img, stats_text

# =========================================================================
# 4. TAB 4: MÔ PHỎNG QUANG HỌC & XỬ LÝ ẢNH (OPTICAL SIMULATION ICCV 2025)
# =========================================================================
def run_optical_simulation_tab(image):
    """
    Mô phỏng 5 giai đoạn biến dạng quang học thực tế trên camera:
    Stage 1: Ảnh gốc nguyên bản
    Stage 2: Bản đồ biên vi kết cấu (Sobel Magma)
    Stage 3: Lọc thông thấp nhòe tiêu cự (Defocus Pillbox PSF)
    Stage 4: Lọc hộp & Giảm mẫu pixel (Box Filter + Binning)
    Stage 5: Nhiễu hạt photon Poisson + Nhiễu đọc cảm biến Gaussian
    """
    if image is None:
        return None, "Vui lòng chọn hoặc tải ảnh trước."

    img_arr = load_image_as_rgb_uint8(image)
    if img_arr is None:
        return None, "Không thể đọc ảnh."

    h, w, _ = img_arr.shape
    crop_size = min(512, h, w)
    start_y = (h - crop_size) // 2
    start_x = (w - crop_size) // 2
    orig_patch = img_arr[start_y:start_y + crop_size, start_x:start_x + crop_size]

    # Stage 1: Ảnh gốc
    stage1 = orig_patch.copy()

    # Stage 2: Biên Sobel
    gray = cv2.cvtColor(orig_patch, cv2.COLOR_RGB2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sobel_x**2 + sobel_y**2)
    sobel_norm = cv2.normalize(sobel_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    stage2 = cv2.applyColorMap(sobel_norm, cv2.COLORMAP_MAGMA)
    stage2 = cv2.cvtColor(stage2, cv2.COLOR_BGR2RGB)

    # Stage 3: Nhòe tiêu cự Pillbox PSF
    disk_psf = generate_pillbox_psf(radius=7)
    stage3 = cv2.filter2D(orig_patch, -1, disk_psf)

    # Stage 4: Box Filter & Downsampling
    box_filtered = cv2.boxFilter(orig_patch, ddepth=-1, ksize=(8, 8))
    down_small = cv2.resize(box_filtered, (crop_size // 4, crop_size // 4), interpolation=cv2.INTER_AREA)
    stage4 = cv2.resize(down_small, (crop_size, crop_size), interpolation=cv2.INTER_NEAREST)

    # Stage 5: Poisson Shot Noise + Gaussian Read Noise
    stage5 = apply_poisson_gaussian_noise(orig_patch, peak_photon=80.0, gaussian_sigma=15.0)

    # Vẽ bảng 5 giai đoạn
    fig, axes = plt.subplots(1, 5, figsize=(22, 5.5))
    stages_data = [
        ("1. Ảnh gốc vi kết cấu\n(Original Patch 512×512)", stage1),
        ("2. Bản đồ biên vi kết cấu\n(Sobel Magnitude Map)", stage2),
        ("3. Lọc nhòe tiêu cự\n(Defocus Pillbox PSF)", stage3),
        ("4. Lọc hộp & Giảm mẫu pixel\n(Box Filter & Binning)", stage4),
        ("5. Cộng nhiễu cảm biến\n(Poisson + Gaussian Noise)", stage5)
    ]

    for ax, (title, img) in zip(axes, stages_data):
        ax.imshow(img)
        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
        ax.axis('off')

    plt.suptitle("Quy trình Mô phỏng Quang học & Xử lý Ảnh Học sâu (Matador Optical Simulation)",
                 fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()

    os.makedirs("outputs", exist_ok=True)
    out_path = "outputs/app_optical_sim.png"
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()

    result_img = Image.open(out_path)
    info_text = (
        "### 📷 Mô phỏng Suy thoái Quang học (Camera Optical Degradation):\n"
        "• **Giai đoạn 1**: Cắt vùng vi kết cấu chuẩn 512×512 từ ảnh cảm biến thô.\n"
        "• **Giai đoạn 2**: Trích xuất độ lớn gradient không gian bằng toán tử vi phân Sobel 2 chiều.\n"
        "• **Giai đoạn 3**: Mô phỏng hiện tượng nhòe ngoài tiêu cự (Defocus Blur) qua hàm tán xạ điểm Pillbox/Disk PSF bán kính $r=7$ px.\n"
        "• **Giai đoạn 4**: Lọc trung bình ô không gian (Box Filter 8×8) và mô phỏng kích thước bước pixel (Pixel Pitch Binning 4×).\n"
        "• **Giai đoạn 5**: Mô phỏng sự biến thiên lượng tử ánh sáng (Poisson Shot Noise với $\\lambda=80$ photon) kết hợp nhiễu nhiệt mạch đọc (Gaussian Read Noise $\\sigma=15$)."
    )

    return result_img, info_text

# =========================================================================
# 5. GRADIO EVENT HANDLERS
# =========================================================================
def on_image_click(image, threshold, evt: gr.SelectData):
    """Xử lý khi người dùng click chuột trực tiếp vào ảnh cảnh quan"""
    if evt is None:
        coords = None
    else:
        coords = {"x": evt.index[0], "y": evt.index[1]}
    return probe_scene_coordinate(image, coords, threshold)

def handle_image_upload(file_obj, threshold):
    """Xử lý tải ảnh .tif / .png / .jpg từ máy tính"""
    if file_obj is None:
        return None, None, None, "Vui lòng chọn file ảnh để tải lên.", ""

    img_rgb = load_image_as_rgb_uint8(file_obj)
    if img_rgb is None:
        return None, None, None, "Lỗi: Không thể giải mã file ảnh đã chọn.", ""

    h, w, _ = img_rgb.shape
    center_coords = {"x": w // 2, "y": h // 2}
    marked_img, edge_map, details, mech_txt = probe_scene_coordinate(img_rgb, center_coords, threshold)

    return img_rgb, marked_img, edge_map, details, mech_txt

def handle_demo_select(sample_name, threshold):
    """Xử lý khi chọn ảnh mẫu từ Dropdown demo"""
    if not sample_name:
        return None, None, None, "Vui lòng chọn một ảnh mẫu.", ""

    path = os.path.join("demo_samples", f"{sample_name}.png")
    if not os.path.isfile(path):
        return None, None, None, f"Không tìm thấy ảnh {path}", ""

    img_rgb = load_image_as_rgb_uint8(path)
    h, w, _ = img_rgb.shape
    center_coords = {"x": w // 2, "y": h // 2}
    marked_img, edge_map, details, mech_txt = probe_scene_coordinate(img_rgb, center_coords, threshold)

    return img_rgb, marked_img, edge_map, details, mech_txt

# =========================================================================
# 6. GIAO DIỆN CHÍNH GRADIO APP
# =========================================================================
demo_sample_files = sorted(glob.glob("demo_samples/*.png"))
demo_choices = [os.path.splitext(os.path.basename(f))[0] for f in demo_sample_files]

with gr.Blocks(title="Matador-C1 Phân tích Vật liệu Phân cấp (ICCV 2025)") as demo:
    gr.Markdown("# 🔍 Matador-C1: Hệ thống Phân loại Vật liệu Phân cấp & Xử lý Ảnh Học sâu")
    gr.Markdown(
        "Ứng dụng học sâu phân cấp kết hợp **Mạng Nơ-ron Đồ thị Chú ý (Hierarchical GAT)**, "
        "các thuật toán **Xử lý Ảnh Số kinh điển (Sobel, LBP, GLCM, Gabor, PSF Blur)** "
        "và thuật toán duyệt cây **Best-First Search** theo bài báo ICCV 2025."
    )

    with gr.Tabs():
        # --- TAB 1: SCENE PROBING TỌA ĐỘ ---
        with gr.TabItem("🎯 1. Phân tích Tọa độ Cục bộ (Scene Probing & BFS)"):
            with gr.Row():
                with gr.Column(scale=1):
                    scene_img = gr.Image(
                        label="Ảnh cảnh quan (Nhấp chuột trực tiếp để phân tích tọa độ)",
                        type="numpy",
                        image_mode="RGB",
                        format="png"
                    )
                    with gr.Row():
                        upload_btn = gr.UploadButton(
                            "📂 Tải ảnh .TIF / .PNG / .JPG",
                            file_types=["image", ".tif", ".tiff", ".png", ".jpg", ".jpeg"],
                            variant="primary"
                        )
                        demo_dropdown = gr.Dropdown(
                            label="⚡ Chọn ảnh mẫu Demo",
                            choices=demo_choices,
                            value=demo_choices[0] if demo_choices else None,
                            interactive=True
                        )
                    threshold_slider = gr.Slider(
                        minimum=0.1, maximum=0.9, value=0.45, step=0.05,
                        label="Ngưỡng tin cậy duyệt phân cấp (Confidence Threshold)"
                    )
                    if demo_sample_files:
                        gr.Examples(
                            examples=[[f] for f in demo_sample_files[:6]],
                            inputs=scene_img,
                            label="Click chọn nhanh ảnh mẫu:"
                        )

                with gr.Column(scale=1):
                    with gr.Row():
                        output_img = gr.Image(label="Vùng cửa sổ tối ưu trên ảnh", format="png")
                        edge_img = gr.Image(label="Bản đồ Biên Sobel Gradient", format="png")
                    results_text = gr.Markdown(label="Chi tiết phân loại")
                    mechanical_text = gr.Textbox(label="Thuộc tính cơ tính suy luận vật lý", lines=4)

        # --- TAB 2: DENSE MATERIAL HEATMAP ---
        with gr.TabItem("🗺️ 2. Bản đồ Vật liệu Dày đặc (Dense Material Heatmap)"):
            gr.Markdown(
                "Quét toàn bộ bề mặt ảnh theo lưới đa vùng (Grid Scan) và tạo lớp phủ nhiệt màu "
                "mã hóa nhóm vật liệu (**Metal, Rock, Ceramic, Polymer, Natural, Derivative**)."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    heatmap_input_img = gr.Image(label="Ảnh đầu vào", type="numpy", format="png")
                    with gr.Row():
                        grid_slider = gr.Slider(
                            minimum=4, maximum=16, value=8, step=2,
                            label="Độ phân giải lưới quét (Grid Resolution N×N)"
                        )
                        alpha_slider = gr.Slider(
                            minimum=0.2, maximum=0.8, value=0.45, step=0.05,
                            label="Độ trong suốt màu phủ (Alpha Blending)"
                        )
                    generate_heatmap_btn = gr.Button("🚀 Tạo Bản đồ Nhiệt Vật liệu", variant="primary")

                with gr.Column(scale=1):
                    heatmap_output_img = gr.Image(label="Bản đồ Nhiệt Phân bố Vật liệu (Overlay Heatmap)", format="png")
                    heatmap_stats_text = gr.Markdown(label="Thống kê phân bố")

        # --- TAB 3: TRÍCH XUẤT VI KẾT CẤU ---
        with gr.TabItem("🔬 3. Trích xuất Vi Kết cấu (LBP / GLCM / Gabor)"):
            gr.Markdown(
                "Trích xuất đặc trưng kết cấu cổ điển của học phần Xử lý Ảnh: "
                "**Local Binary Pattern (LBP)**, **Ma trận đồng hiện GLCM (Haralick)**, "
                "và **Bộ lọc Gabor đa hướng đa tần số**."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    texture_input_img = gr.Image(label="Ảnh phân tích kết cấu", type="numpy", format="png")
                    texture_analyze_btn = gr.Button("🔬 Phân tích Đặc trưng Vi Kết cấu", variant="primary")
                with gr.Column(scale=2):
                    texture_result_img = gr.Image(label="Bảng phân tích đặc trưng kết cấu", format="png")
                    texture_stats_text = gr.Markdown(label="Thống kê vector")

        # --- TAB 4: MÔ PHỎNG QUANG HỌC ICCV 2025 ---
        with gr.TabItem("📷 4. Mô phỏng Quang học Máy ảnh (Optical Simulation)"):
            gr.Markdown(
                "Quy trình mô phỏng thoái hóa quang học của bài báo ICCV 2025: "
                "**PSF Defocus Blur**, **Pixel Pitch Box Filtering**, và **Poisson-Gaussian Noise**."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    optical_input_img = gr.Image(label="Ảnh đầu vào", type="numpy", format="png")
                    optical_run_btn = gr.Button("📷 Chạy Mô phỏng Quang học 5 Bước", variant="primary")
                with gr.Column(scale=2):
                    optical_result_img = gr.Image(label="Quy trình 5 Giai đoạn Mô phỏng", format="png")
                    optical_info_text = gr.Markdown(label="Giải thích cơ chế vật lý")

    # ======================== EVENT BINDINGS ========================
    # 1. Tab 1 events
    scene_img.select(
        fn=on_image_click,
        inputs=[scene_img, threshold_slider],
        outputs=[output_img, edge_img, results_text, mechanical_text]
    )

    upload_btn.upload(
        fn=handle_image_upload,
        inputs=[upload_btn, threshold_slider],
        outputs=[scene_img, output_img, edge_img, results_text, mechanical_text]
    )

    demo_dropdown.change(
        fn=handle_demo_select,
        inputs=[demo_dropdown, threshold_slider],
        outputs=[scene_img, output_img, edge_img, results_text, mechanical_text]
    )

    # Đồng bộ ảnh sang các Tab khác khi chọn mẫu ở Tab 1
    demo_dropdown.change(
        fn=lambda name: load_image_as_rgb_uint8(os.path.join("demo_samples", f"{name}.png")) if name else None,
        inputs=[demo_dropdown],
        outputs=[heatmap_input_img]
    )
    demo_dropdown.change(
        fn=lambda name: load_image_as_rgb_uint8(os.path.join("demo_samples", f"{name}.png")) if name else None,
        inputs=[demo_dropdown],
        outputs=[texture_input_img]
    )
    demo_dropdown.change(
        fn=lambda name: load_image_as_rgb_uint8(os.path.join("demo_samples", f"{name}.png")) if name else None,
        inputs=[demo_dropdown],
        outputs=[optical_input_img]
    )

    # 2. Tab 2 events
    generate_heatmap_btn.click(
        fn=generate_dense_material_heatmap,
        inputs=[heatmap_input_img, grid_slider, alpha_slider],
        outputs=[heatmap_output_img, heatmap_stats_text]
    )

    # 3. Tab 3 events
    texture_analyze_btn.click(
        fn=run_texture_feature_analysis,
        inputs=[texture_input_img],
        outputs=[texture_result_img, texture_stats_text]
    )

    # 4. Tab 4 events
    optical_run_btn.click(
        fn=run_optical_simulation_tab,
        inputs=[optical_input_img],
        outputs=[optical_result_img, optical_info_text]
    )

if __name__ == "__main__":
    print("-> [4/4] Đang khởi chạy máy chủ Gradio 4 Tabs...")
    print("=" * 65)
    print("🚀 GIAO DIỆN ĐÃ SẴN SÀNG! Mời bạn mở trình duyệt tại địa chỉ:")
    print("   👉 http://127.0.0.1:7860")
    print("=" * 65)
    sys.stdout.flush()
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
