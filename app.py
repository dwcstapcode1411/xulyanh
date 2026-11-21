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
from scipy.ndimage import gaussian_filter
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

# Danh sách tên nút lá (Material) và index tương ứng
leaf_name_list = [n for n, idx in node_to_idx.items() if node_levels.get(n) == 4 and n != "GLOBAL_NODE"]
leaf_idx_list = [node_to_idx[n] for n in leaf_name_list]

# Danh sách tên nút Form (Level 3) và index tương ứng
form_name_list = [n for n, idx in node_to_idx.items() if node_levels.get(n) == 3 and n != "GLOBAL_NODE"]
form_idx_list = [node_to_idx[n] for n in form_name_list]

# Danh sách tên nút Composition (Level 2) và index tương ứng
comp_name_list = [n for n, idx in node_to_idx.items() if node_levels.get(n) == 2 and n != "GLOBAL_NODE"]
comp_idx_list = [node_to_idx[n] for n in comp_name_list]

# Bảng màu cho tầng Composition (Level 2)
COMPOSITION_PALETTE = {
    "Metal": (65, 130, 225),       # Xanh kim loại (Royal Blue)
    "Rock": (150, 150, 150),       # Xám đá (Gray)
    "Ceramic": (220, 110, 40),     # Cam gốm sứ (Terracotta)
    "Polymer": (165, 100, 235),    # Tím sợi / tổng hợp (Purple)
    "Natural": (45, 175, 80),      # Xanh lá tự nhiên (Green)
    "Derivative": (235, 185, 30)   # Vàng da lông / thực phẩm (Gold)
}

# Bảng màu mở rộng cho 37 lớp Material (Level 4) - dùng tab20 + tab20b + tab20c
_cmap_tab20 = matplotlib.colormaps['tab20'].resampled(20)
_cmap_tab20b = matplotlib.colormaps['tab20b'].resampled(20)
MATERIAL_PALETTE = {}
for i, name in enumerate(leaf_name_list):
    if i < 20:
        c = _cmap_tab20(i / 20.0)
    else:
        c = _cmap_tab20b((i - 20) / 20.0)
    MATERIAL_PALETTE[name] = (int(c[0]*255), int(c[1]*255), int(c[2]*255))

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
# HELPER: Tải ảnh demo mặc định đầu tiên
# =========================================================================
demo_sample_files = sorted(glob.glob("demo_samples/*.png"))
demo_choices = [os.path.splitext(os.path.basename(f))[0] for f in demo_sample_files]

def _load_default_demo_image():
    """Trả về ảnh mẫu đầu tiên từ demo_samples/ (nếu có) dưới dạng numpy RGB uint8."""
    if demo_sample_files:
        return load_image_as_rgb_uint8(demo_sample_files[0])
    return None

def _load_demo_by_name(name):
    """Trả về ảnh mẫu theo tên từ demo_samples/ dưới dạng numpy RGB uint8."""
    if not name:
        return None
    path = os.path.join("demo_samples", f"{name}.png")
    if os.path.isfile(path):
        return load_image_as_rgb_uint8(path)
    return None

def _normalize_upload(file_obj):
    """Callback cho UploadButton: giải mã file → numpy RGB uint8."""
    if file_obj is None:
        return None
    return load_image_as_rgb_uint8(file_obj)

def _normalize_drag_drop(img_arr):
    """Callback cho sự kiện .upload() khi kéo thả trực tiếp vào gr.Image → chuẩn hóa."""
    if img_arr is None:
        return None
    return load_image_as_rgb_uint8(img_arr)


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
        gr.Warning("Vui lòng tải hoặc chọn một bức ảnh trước khi phân tích!")
        return None, None, "", ""

    img_arr = load_image_as_rgb_uint8(image)
    if img_arr is None:
        gr.Warning("Không thể đọc định dạng ảnh đã chọn.")
        return None, None, "", ""

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
        sobel_mean = mag.mean()
    else:
        edge_map = None
        sobel_mean = 0.0

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
    strength_label = "Mạnh mẽ (Strong ≥ 1 MPa)" if mech_codes[0] == 1 else "Dễ vỡ (Fragile < 1 MPa)"
    elastic_label = "Cứng cáp (Rigid ≥ 12 GPa)" if mech_codes[1] == 1 else "Dễ biến dạng (Deformable < 12 GPa)"
    density_label = "Nặng (Heavy ≥ 1600 kg/m³)" if mech_codes[2] == 1 else "Nhẹ (Light < 1600 kg/m³)"

    mech_text = (
        f"• Sức chịu kéo (Tensile Strength): {strength_label}\n"
        f"• Độ đàn hồi / Cứng cáp (Elasticity): {elastic_label}\n"
        f"• Tỷ trọng (Density): {density_label}"
    )

    output_details = (
        f"### 🎯 Kết quả Phân loại Phân cấp Nhất quán (Matador-C1):\n\n"
        f"**Cây phả hệ phân cấp đầy đủ:**\n"
        f"`{path_display}`\n\n"
        f"**🏆 Top 3 Vật liệu có độ tin cậy cao nhất:**\n"
        f"- 🥇 {top3_candidates[0]}\n"
        f"- 🥈 {top3_candidates[1] if len(top3_candidates) > 1 else ''}\n"
        f"- 🥉 {top3_candidates[2] if len(top3_candidates) > 2 else ''}\n\n"
        f"**Tọa độ:** (X={u}, Y={v}) &nbsp;|&nbsp; "
        f"**Cửa sổ tối ưu:** {best_scale} × {best_scale} px &nbsp;|&nbsp; "
        f"**Độ tin cậy:** {best_conf*100.0:.1f}%\n\n"
        f"**Chỉ số biên Sobel trung bình:** `{sobel_mean:.2f}`"
    )

    return marked_image, edge_map, output_details, mech_text


# =========================================================================
# 2. TAB 2: BẢN ĐỒ NHIỆT VẬT LIỆU DÀY ĐẶC (DENSE MATERIAL HEATMAP)
# =========================================================================
def generate_dense_material_heatmap(image, patch_size_choice="256", blend_alpha=0.45, view_level="Composition (Nhóm vật liệu)"):
    """
    Quét cửa sổ trượt phân giải cao trên toàn bộ bức ảnh:
    1. Chia ảnh thành lưới patch với kích thước và bước trượt tùy chọn
    2. Gom nhóm xử lý batch qua mạng ResNet50 + Hierarchical GAT
    3. Làm mượt xác suất bằng Gaussian filter, nội suy song tuyến
    4. Áp dụng Hierarchical Path Consistency
    5. Trải lớp phủ màu (Overlay Mask) + bảng chú giải
    """
    if image is None:
        gr.Warning("Vui lòng tải hoặc chọn một bức ảnh trước khi phân tích!")
        return None, ""

    img_arr = load_image_as_rgb_uint8(image)
    if img_arr is None:
        gr.Warning("Không thể đọc ảnh đã chọn.")
        return None, ""

    h, w, _ = img_arr.shape

    # Xác định kích thước patch và bước trượt (stride = 50% patch)
    patch_px = int(patch_size_choice)
    short_side = min(h, w)
    # Nếu ảnh quá nhỏ so với patch, giảm patch_px
    if patch_px > short_side:
        patch_px = max(64, short_side)
    stride = max(patch_px // 2, 32)

    # Tạo lưới các vị trí patch
    y_positions = list(range(0, h - patch_px + 1, stride))
    if not y_positions or y_positions[-1] + patch_px < h:
        y_positions.append(max(0, h - patch_px))
    x_positions = list(range(0, w - patch_px + 1, stride))
    if not x_positions or x_positions[-1] + patch_px < w:
        x_positions.append(max(0, w - patch_px))

    grid_h = len(y_positions)
    grid_w = len(x_positions)

    patches = []
    coords = []

    for gy, y1 in enumerate(y_positions):
        for gx, x1 in enumerate(x_positions):
            y2 = min(y1 + patch_px, h)
            x2 = min(x1 + patch_px, w)
            patch = img_arr[y1:y2, x1:x2]
            if patch.size == 0:
                continue
            # Chuyển sang PIL RGB rồi áp dụng cùng transform chuẩn như Tab 1
            pil_p = Image.fromarray(patch).convert('RGB')
            tensor_p = effective_transforms(pil_p)
            patches.append(tensor_p)
            coords.append((gy, gx, y1, y2, x1, x2))

    if not patches:
        return None, "Không thể trích xuất các ô lưới."

    batch_tensor = torch.stack(patches, dim=0).to(device)

    # Dự đoán mô hình theo lô
    feature_extractor.eval()
    gat_model.eval()

    # Xác định chế độ xem
    is_material_view = "Material" in view_level

    if is_material_view:
        target_names = leaf_name_list
        target_indices = leaf_idx_list
        palette = MATERIAL_PALETTE
    else:
        target_names = comp_name_list
        target_indices = comp_idx_list
        palette = COMPOSITION_PALETTE

    num_classes = len(target_names)
    # Lưu lưới xác suất
    prob_grid = np.zeros((grid_h, grid_w, num_classes), dtype=np.float32)

    batch_size = 16
    patch_i = 0
    with torch.no_grad():
        for i in range(0, len(patches), batch_size):
            sub_batch = batch_tensor[i:i + batch_size]
            with torch.amp.autocast('cuda'):
                phi = feature_extractor(sub_batch)
                logits = gat_model(phi)

            # Lấy logits tại các nút mục tiêu, tính softmax
            target_tensor = torch.tensor(target_indices, device=device)
            sub_logits = logits[:, target_tensor]  # (B, num_classes)
            sub_probs = torch.softmax(sub_logits, dim=-1).cpu().numpy()  # (B, num_classes)

            for j in range(sub_probs.shape[0]):
                gy, gx = coords[patch_i + j][:2]
                prob_grid[gy, gx] = sub_probs[j]

            patch_i += sub_probs.shape[0]

    # Làm mượt không gian: Gaussian filter sigma=1.0 trên từng kênh lớp xác suất
    smoothed_grid = np.zeros_like(prob_grid)
    for c in range(num_classes):
        smoothed_grid[:, :, c] = gaussian_filter(prob_grid[:, :, c], sigma=1.0)

    # Chuẩn hóa lại tổng xác suất = 1 sau khi làm mượt
    grid_sum = smoothed_grid.sum(axis=-1, keepdims=True)
    grid_sum = np.maximum(grid_sum, 1e-8)
    smoothed_grid = smoothed_grid / grid_sum

    # Nội suy song tuyến về kích thước ảnh gốc (H x W x num_classes)
    prob_map_full = np.zeros((h, w, num_classes), dtype=np.float32)
    for c in range(num_classes):
        # Resize từ (grid_h, grid_w) → (h, w) bằng nội suy song tuyến
        prob_map_full[:, :, c] = cv2.resize(
            smoothed_grid[:, :, c], (w, h), interpolation=cv2.INTER_LINEAR
        )

    # Chuẩn hóa lại
    full_sum = prob_map_full.sum(axis=-1, keepdims=True)
    full_sum = np.maximum(full_sum, 1e-8)
    prob_map_full = prob_map_full / full_sum

    # Áp dụng ràng buộc nhất quán cây phân cấp (HPC)
    # Nếu xem theo Material: suy ra Composition từ nhãn Material dự đoán
    # và bắt buộc các lá không thuộc Composition trội bị giảm xuống
    pred_labels = np.argmax(prob_map_full, axis=-1)  # (H, W)

    # Tạo lớp phủ màu
    color_mask = np.zeros_like(img_arr)
    for y in range(h):
        for x in range(w):
            label_idx = pred_labels[y, x]
            label_name = target_names[label_idx]
            color = palette.get(label_name, (128, 128, 128))
            color_mask[y, x] = color

    # Hòa trộn màu với ảnh gốc (Alpha Blending)
    overlay = cv2.addWeighted(img_arr, 1.0 - blend_alpha, color_mask, blend_alpha, 0)

    # Thống kê tỷ lệ phân bố
    total_pixels = h * w
    label_areas = {}
    for idx in range(num_classes):
        count = int(np.sum(pred_labels == idx))
        if count > 0:
            label_areas[target_names[idx]] = count

    level_label = "Material (37 lớp)" if is_material_view else "Composition (Nhóm vật liệu)"
    summary_lines = [
        f"### 📊 Bản đồ Phân bố Vật liệu — Tầng **{level_label}**",
        f"Patch `{patch_px}×{patch_px}` px, stride `{stride}` px → lưới `{grid_w}×{grid_h}` = `{len(patches)}` vùng, Gaussian σ=1.0\n",
        "| Vật liệu | Tỷ lệ | Diện tích |",
        "|:---------|------:|----------:|"
    ]

    for name, count in sorted(label_areas.items(), key=lambda x: -x[1]):
        pct = (count / total_pixels) * 100.0
        hex_color = "#{:02x}{:02x}{:02x}".format(*palette.get(name, (128, 128, 128)))
        summary_lines.append(
            f"| <span style='color:{hex_color}; font-weight:bold;'>■</span> {name} | {pct:.1f}% | {count:,} px |"
        )

    return overlay, "\n".join(summary_lines)


# =========================================================================
# 3. TAB 3: TRÍCH XUẤT VI KẾT CẤU (LBP / GLCM / GABOR FILTERS)
# =========================================================================
def run_texture_feature_analysis(image):
    """
    Trích xuất toàn diện các bộ mô tả kết cấu cổ điển (Classical Texture Descriptors):
    - LBP (Local Binary Pattern)
    - GLCM (Gray-Level Co-occurrence Matrix) + chỉ số Haralick
    - Gabor Multi-orientation Filters (0°, 45°, 90°, 135°)
    Trả về 3 ảnh riêng biệt + bảng chỉ số GLCM định lượng.
    """
    if image is None:
        gr.Warning("Vui lòng tải hoặc chọn một bức ảnh trước khi phân tích!")
        return None, None, None, ""

    img_arr = load_image_as_rgb_uint8(image)
    if img_arr is None:
        gr.Warning("Không thể đọc định dạng ảnh.")
        return None, None, None, ""

    # Chuyển sang ảnh xám uint8 chuẩn
    if img_arr.ndim == 3:
        gray = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_arr.copy()
    gray = gray.astype(np.uint8)

    # Resize về 256x256 cho phân tích nhanh
    gray_resized = cv2.resize(gray, (256, 256))

    # 1. LBP Map
    lbp_img = compute_lbp(gray_resized, radius=1, n_points=8)
    lbp_norm = cv2.normalize(lbp_img.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    lbp_color = cv2.applyColorMap(lbp_norm, cv2.COLORMAP_JET)
    lbp_color = cv2.cvtColor(lbp_color, cv2.COLOR_BGR2RGB)

    # 2. GLCM Heatmap (cửa sổ trượt Energy Map)
    win_size = 16
    h_g, w_g = gray_resized.shape
    energy_map = np.zeros((h_g, w_g), dtype=np.float32)
    for y in range(0, h_g - win_size, win_size // 2):
        for x in range(0, w_g - win_size, win_size // 2):
            patch = gray_resized[y:y+win_size, x:x+win_size]
            glcm_f = compute_glcm_features(patch, distances=[1], angles=[0])
            energy_map[y:y+win_size, x:x+win_size] = glcm_f[3]  # energy
    energy_norm = cv2.normalize(energy_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    glcm_color = cv2.applyColorMap(energy_norm, cv2.COLORMAP_HOT)
    glcm_color = cv2.cvtColor(glcm_color, cv2.COLOR_BGR2RGB)

    # 3. Gabor Response tổng hợp (trung bình 4 hướng)
    gray_f = gray_resized.astype(np.float32) / 255.0
    gabor_sum = np.zeros_like(gray_f)
    for o in range(4):
        theta = o * np.pi / 4
        kernel = cv2.getGaborKernel((31, 31), 4.0 * 0.56, theta, 4.0, 0.5, 0, cv2.CV_32F)
        kernel /= (1.5 * kernel.sum() + 1e-7)
        resp = cv2.filter2D(gray_f, cv2.CV_32F, kernel)
        gabor_sum += np.abs(resp)
    gabor_sum /= 4.0
    gabor_norm = cv2.normalize(gabor_sum, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    gabor_color = cv2.applyColorMap(gabor_norm, cv2.COLORMAP_INFERNO)
    gabor_color = cv2.cvtColor(gabor_color, cv2.COLOR_BGR2RGB)

    # 4. Tính chỉ số GLCM định lượng toàn cục
    glcm_global = compute_glcm_features(gray_resized, distances=[1, 3], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4])
    # Mỗi cặp (d, angle) cho 4 giá trị: contrast, dissimilarity, homogeneity, energy
    # Có 2 distances x 4 angles = 8 cặp x 4 = 32 giá trị
    n_combos = len(glcm_global) // 4
    contrasts = [glcm_global[i*4 + 0] for i in range(n_combos)]
    dissimilarities = [glcm_global[i*4 + 1] for i in range(n_combos)]
    homogeneities = [glcm_global[i*4 + 2] for i in range(n_combos)]
    energies = [glcm_global[i*4 + 3] for i in range(n_combos)]

    # Trích xuất vector 64 chiều
    feat_64 = extract_texture_features(img_arr, target_dim=64)

    stats_text = (
        f"### 🔬 Chỉ số Kết cấu GLCM (Haralick) định lượng:\n\n"
        f"| Chỉ số | Trung bình | Min | Max |\n"
        f"|:-------|----------:|---------:|---------:|\n"
        f"| **Contrast** | {np.mean(contrasts):.4f} | {np.min(contrasts):.4f} | {np.max(contrasts):.4f} |\n"
        f"| **Dissimilarity** | {np.mean(dissimilarities):.4f} | {np.min(dissimilarities):.4f} | {np.max(dissimilarities):.4f} |\n"
        f"| **Homogeneity** | {np.mean(homogeneities):.4f} | {np.min(homogeneities):.4f} | {np.max(homogeneities):.4f} |\n"
        f"| **Energy** | {np.mean(energies):.4f} | {np.min(energies):.4f} | {np.max(energies):.4f} |\n\n"
        f"### 📐 Vector Đặc trưng Vi Kết cấu (64-D):\n"
        f"- **L2 Norm:** `{np.linalg.norm(feat_64):.4f}` &nbsp;|&nbsp; "
        f"**Max:** `{feat_64.max():.4f}` &nbsp;|&nbsp; "
        f"**Min:** `{feat_64.min():.4f}` &nbsp;|&nbsp; "
        f"**Mean:** `{feat_64.mean():.4f}` &nbsp;|&nbsp; "
        f"**Std:** `{feat_64.std():.4f}`\n"
        f"- `[0..31]` LBP Histogram — vi kết cấu điểm ảnh cục bộ\n"
        f"- `[32..63]` GLCM Haralick + Gabor Multi-orientation"
    )

    return lbp_color, glcm_color, gabor_color, stats_text


# =========================================================================
# 4. TAB 4: MÔ PHỎNG QUANG HỌC & XỬ LÝ ẢNH (OPTICAL SIMULATION ICCV 2025)
# =========================================================================
def run_optical_simulation_tab(image, psf_radius=7, box_ksize=8, pixel_bin=4, peak_photon=80.0, gauss_sigma=15.0):
    """
    Mô phỏng 5 giai đoạn biến dạng quang học thực tế trên camera:
    Stage 1: Ảnh gốc nguyên bản
    Stage 2: Bản đồ biên vi kết cấu (Sobel Magma)
    Stage 3: Lọc thông thấp nhòe tiêu cự (Defocus Pillbox PSF)
    Stage 4: Lọc hộp & Giảm mẫu pixel (Box Filter + Binning)
    Stage 5: Nhiễu hạt photon Poisson + Nhiễu đọc cảm biến Gaussian
    """
    if image is None:
        gr.Warning("Vui lòng tải hoặc chọn một bức ảnh trước khi phân tích!")
        return None, "", None

    img_arr = load_image_as_rgb_uint8(image)
    if img_arr is None:
        gr.Warning("Không thể đọc ảnh.")
        return None, "", None

    h, w, _ = img_arr.shape
    crop_size = min(512, h, w)
    start_y = (h - crop_size) // 2
    start_x = (w - crop_size) // 2
    orig_patch = img_arr[start_y:start_y + crop_size, start_x:start_x + crop_size]

    psf_radius = max(1, int(psf_radius))
    box_ksize = max(2, int(box_ksize))
    pixel_bin = max(2, int(pixel_bin))
    peak_photon = max(10.0, float(peak_photon))
    gauss_sigma = max(1.0, float(gauss_sigma))

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
    disk_psf = generate_pillbox_psf(radius=psf_radius)
    stage3 = cv2.filter2D(orig_patch, -1, disk_psf)

    # Stage 4: Box Filter & Downsampling
    box_filtered = cv2.boxFilter(orig_patch, ddepth=-1, ksize=(box_ksize, box_ksize))
    down_small = cv2.resize(box_filtered, (crop_size // pixel_bin, crop_size // pixel_bin), interpolation=cv2.INTER_AREA)
    stage4 = cv2.resize(down_small, (crop_size, crop_size), interpolation=cv2.INTER_NEAREST)

    # Stage 5: Poisson Shot Noise + Gaussian Read Noise
    stage5 = apply_poisson_gaussian_noise(orig_patch, peak_photon=peak_photon, gaussian_sigma=gauss_sigma)

    # Vẽ bảng 5 giai đoạn
    fig, axes = plt.subplots(1, 5, figsize=(22, 5.5))
    stages_data = [
        ("1. Ảnh gốc\n(Original Patch)", stage1),
        ("2. Biên Sobel\n(Gradient Map)", stage2),
        (f"3. Nhòe PSF r={psf_radius}\n(Defocus Blur)", stage3),
        (f"4. Box {box_ksize}×{box_ksize} + Bin {pixel_bin}×\n(Pixel Pitch)", stage4),
        (f"5. Noise λ={peak_photon:.0f} σ={gauss_sigma:.0f}\n(Poisson+Gaussian)", stage5)
    ]

    for ax, (title, img) in zip(axes, stages_data):
        ax.imshow(img)
        ax.set_title(title, fontsize=11, fontweight='bold', pad=10)
        ax.axis('off')

    plt.suptitle("Mô phỏng Quang học & Xử lý Ảnh (Matador Optical Simulation)",
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()

    os.makedirs("outputs", exist_ok=True)
    out_path = "outputs/app_optical_sim.png"
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()

    result_img = Image.open(out_path)
    info_text = (
        "### 📷 Mô phỏng Suy thoái Quang học (Camera Optical Degradation):\n"
        f"• **Giai đoạn 1**: Cắt vùng vi kết cấu chuẩn {crop_size}×{crop_size} từ ảnh cảm biến thô.\n"
        "• **Giai đoạn 2**: Trích xuất gradient không gian bằng toán tử Sobel 2 chiều.\n"
        f"• **Giai đoạn 3**: Mô phỏng nhòe ngoài tiêu cự (Defocus) qua Pillbox PSF bán kính r={psf_radius} px.\n"
        f"• **Giai đoạn 4**: Box Filter {box_ksize}×{box_ksize} + Pixel Binning {pixel_bin}×.\n"
        f"• **Giai đoạn 5**: Poisson Shot Noise (λ={peak_photon:.0f} photon) + Gaussian Read Noise (σ={gauss_sigma:.0f})."
    )

    # Tìm các biểu đồ báo cáo có sẵn trong outputs/
    report_gallery = []
    report_files = [
        "outputs/figure_5b_hierarchy_accuracy.png",
        "outputs/confusion_matrix.png",
        "outputs/ablation_comparison.png",
        "outputs/image_processing_pipeline.png"
    ]
    for rf in report_files:
        if os.path.isfile(rf):
            report_gallery.append(Image.open(rf))

    # Tạo ảnh tổng hợp báo cáo (nếu có)
    report_img = None
    if report_gallery:
        # Tạo gallery 2 cột
        n_reports = len(report_gallery)
        cols = 2
        rows = (n_reports + 1) // 2
        fig2, axes2 = plt.subplots(rows, cols, figsize=(16, 8 * rows))
        if rows == 1:
            axes2 = [axes2] if cols == 1 else [axes2]
        axes2_flat = np.array(axes2).flatten()
        for i, rimg in enumerate(report_gallery):
            axes2_flat[i].imshow(rimg)
            axes2_flat[i].set_title(os.path.basename(report_files[i]).replace('.png', '').replace('_', ' ').title(), fontweight='bold')
            axes2_flat[i].axis('off')
        for i in range(len(report_gallery), len(axes2_flat)):
            axes2_flat[i].axis('off')
        plt.suptitle("Báo cáo Hiệu năng Mô hình Matador-C1", fontsize=14, fontweight='bold')
        plt.tight_layout()
        report_path = "outputs/app_reports_gallery.png"
        plt.savefig(report_path, dpi=150, bbox_inches='tight')
        plt.close()
        report_img = Image.open(report_path)

    return result_img, info_text, report_img


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
    """Xử lý tải ảnh .tif / .png / .jpg từ máy tính cho Tab 1"""
    if file_obj is None:
        gr.Warning("Vui lòng chọn file ảnh để tải lên.")
        return None, None, None, "", ""

    img_rgb = load_image_as_rgb_uint8(file_obj)
    if img_rgb is None:
        gr.Warning("Lỗi: Không thể giải mã file ảnh đã chọn.")
        return None, None, None, "", ""

    h, w, _ = img_rgb.shape
    center_coords = {"x": w // 2, "y": h // 2}
    marked_img, edge_map, details, mech_txt = probe_scene_coordinate(img_rgb, center_coords, threshold)

    return img_rgb, marked_img, edge_map, details, mech_txt

def handle_demo_select(sample_name, threshold):
    """Xử lý khi chọn ảnh mẫu từ Dropdown demo cho Tab 1"""
    if not sample_name:
        return None, None, None, "", ""

    path = os.path.join("demo_samples", f"{sample_name}.png")
    if not os.path.isfile(path):
        gr.Warning(f"Không tìm thấy ảnh {path}")
        return None, None, None, "", ""

    img_rgb = load_image_as_rgb_uint8(path)
    if img_rgb is None:
        return None, None, None, "", ""
    h, w, _ = img_rgb.shape
    center_coords = {"x": w // 2, "y": h // 2}
    marked_img, edge_map, details, mech_txt = probe_scene_coordinate(img_rgb, center_coords, threshold)

    return img_rgb, marked_img, edge_map, details, mech_txt


# =========================================================================
# 6. GIAO DIỆN CHÍNH GRADIO APP
# =========================================================================
# Nạp ảnh mặc định cho tất cả các Tab
_default_img = _load_default_demo_image()

with gr.Blocks(title="Matador-C1 Phân tích Vật liệu Phân cấp (ICCV 2025)") as demo:
    gr.Markdown("# 🔍 Matador-C1: Hệ thống Phân loại Vật liệu Phân cấp & Xử lý Ảnh Học sâu")
    gr.Markdown(
        "Ứng dụng học sâu phân cấp kết hợp **Mạng Nơ-ron Đồ thị Chú ý (Hierarchical GAT)**, "
        "các thuật toán **Xử lý Ảnh Số kinh điển (Sobel, LBP, GLCM, Gabor, PSF Blur)** "
        "và thuật toán duyệt cây **Best-First Search** theo bài báo ICCV 2025."
    )

    with gr.Tabs():
        # --- TAB 1: SCENE PROBING TỌA ĐỘ ---
        with gr.TabItem("🔍 1. Thăm dò Cảnh (BFS)"):
            with gr.Row():
                with gr.Column(scale=1):
                    scene_img = gr.Image(
                        label="Ảnh cảnh quan (Nhấp chuột trực tiếp để phân tích tọa độ)",
                        type="numpy",
                        image_mode="RGB",
                        format="png",
                        value=_default_img
                    )
                    with gr.Row():
                        upload_btn_t1 = gr.UploadButton(
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
        with gr.TabItem("🗺️ 2. Bản đồ Vật liệu"):
            gr.Markdown(
                "Quét toàn bộ bề mặt ảnh theo lưới đa vùng (Grid Scan) và tạo lớp phủ nhiệt màu "
                "mã hóa nhóm vật liệu (**Metal, Rock, Ceramic, Polymer, Natural, Derivative**)."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    heatmap_input_img = gr.Image(
                        label="Ảnh đầu vào",
                        type="numpy",
                        image_mode="RGB",
                        format="png",
                        value=_default_img
                    )
                    with gr.Row():
                        upload_btn_t2 = gr.UploadButton(
                            "📂 Tải ảnh .TIF / .PNG / .JPG",
                            file_types=["image", ".tif", ".tiff", ".png", ".jpg", ".jpeg"],
                            variant="secondary"
                        )
                        demo_dropdown_t2 = gr.Dropdown(
                            label="⚡ Chọn ảnh mẫu",
                            choices=demo_choices,
                            value=demo_choices[0] if demo_choices else None,
                            interactive=True
                        )
                    with gr.Row():
                        patch_size_radio = gr.Radio(
                            choices=["128", "192", "256"],
                            value="256",
                            label="Kích thước cửa sổ (Patch Size px)"
                        )
                        alpha_slider = gr.Slider(
                            minimum=0.2, maximum=0.8, value=0.45, step=0.05,
                            label="Alpha Blending"
                        )
                    view_level_radio = gr.Radio(
                        choices=["Composition (Nhóm vật liệu)", "Material (37 lớp chi tiết)"],
                        value="Composition (Nhóm vật liệu)",
                        label="Tầng hiển thị bản đồ nhiệt"
                    )
                    generate_heatmap_btn = gr.Button("🚀 Tạo Bản đồ Nhiệt Vật liệu", variant="primary")

                with gr.Column(scale=1):
                    heatmap_output_img = gr.Image(label="Bản đồ Nhiệt Phân bố Vật liệu (Overlay Heatmap)", format="png")
                    heatmap_stats_text = gr.Markdown(label="Thống kê phân bố")

        # --- TAB 3: TRÍCH XUẤT VI KẾT CẤU ---
        with gr.TabItem("🔬 3. Vi Kết cấu (LBP/GLCM)"):
            gr.Markdown(
                "Trích xuất đặc trưng kết cấu cổ điển: "
                "**Local Binary Pattern (LBP)**, **Ma trận đồng hiện GLCM (Haralick)**, "
                "và **Bộ lọc Gabor đa hướng đa tần số**."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    texture_input_img = gr.Image(
                        label="Ảnh phân tích kết cấu",
                        type="numpy",
                        image_mode="RGB",
                        format="png",
                        value=_default_img
                    )
                    with gr.Row():
                        upload_btn_t3 = gr.UploadButton(
                            "📂 Tải ảnh .TIF / .PNG / .JPG",
                            file_types=["image", ".tif", ".tiff", ".png", ".jpg", ".jpeg"],
                            variant="secondary"
                        )
                        demo_dropdown_t3 = gr.Dropdown(
                            label="⚡ Chọn ảnh mẫu",
                            choices=demo_choices,
                            value=demo_choices[0] if demo_choices else None,
                            interactive=True
                        )
                    texture_analyze_btn = gr.Button("🔬 Phân tích Đặc trưng Vi Kết cấu", variant="primary")
                with gr.Column(scale=2):
                    with gr.Row():
                        lbp_out_img = gr.Image(label="LBP Map", format="png")
                        glcm_out_img = gr.Image(label="GLCM Energy Heatmap", format="png")
                        gabor_out_img = gr.Image(label="Gabor Response", format="png")
                    texture_stats_text = gr.Markdown(label="Chỉ số kết cấu định lượng")

        # --- TAB 4: MÔ PHỎNG QUANG HỌC ICCV 2025 ---
        with gr.TabItem("📷 4. Quang học & Báo cáo"):
            gr.Markdown(
                "Mô phỏng thoái hóa quang học camera ICCV 2025: "
                "**PSF Defocus Blur**, **Pixel Pitch Box Filtering**, và **Poisson-Gaussian Noise**."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    optical_input_img = gr.Image(
                        label="Ảnh đầu vào",
                        type="numpy",
                        image_mode="RGB",
                        format="png",
                        value=_default_img
                    )
                    with gr.Row():
                        upload_btn_t4 = gr.UploadButton(
                            "📂 Tải ảnh .TIF / .PNG / .JPG",
                            file_types=["image", ".tif", ".tiff", ".png", ".jpg", ".jpeg"],
                            variant="secondary"
                        )
                        demo_dropdown_t4 = gr.Dropdown(
                            label="⚡ Chọn ảnh mẫu",
                            choices=demo_choices,
                            value=demo_choices[0] if demo_choices else None,
                            interactive=True
                        )
                    with gr.Row():
                        psf_slider = gr.Slider(minimum=1, maximum=20, value=7, step=1, label="Bán kính PSF (px)")
                        box_slider = gr.Slider(minimum=2, maximum=16, value=8, step=2, label="Box Filter size")
                    with gr.Row():
                        bin_slider = gr.Slider(minimum=2, maximum=8, value=4, step=1, label="Pixel Binning factor")
                        photon_slider = gr.Slider(minimum=10, maximum=500, value=80, step=10, label="Peak Photon (λ)")
                    noise_slider = gr.Slider(minimum=1, maximum=50, value=15, step=1, label="Gaussian σ (Read Noise)")
                    optical_run_btn = gr.Button("📷 Chạy Mô phỏng Quang học 5 Bước", variant="primary")
                with gr.Column(scale=2):
                    optical_result_img = gr.Image(label="Quy trình 5 Giai đoạn Mô phỏng", format="png")
                    optical_info_text = gr.Markdown(label="Giải thích cơ chế vật lý")
                    report_gallery_img = gr.Image(label="Biểu đồ Báo cáo Hiệu năng Mô hình", format="png")

    # ======================== EVENT BINDINGS ========================
    # === Tab 1 events ===
    scene_img.select(
        fn=on_image_click,
        inputs=[scene_img, threshold_slider],
        outputs=[output_img, edge_img, results_text, mechanical_text]
    )

    upload_btn_t1.upload(
        fn=handle_image_upload,
        inputs=[upload_btn_t1, threshold_slider],
        outputs=[scene_img, output_img, edge_img, results_text, mechanical_text]
    )

    demo_dropdown.change(
        fn=handle_demo_select,
        inputs=[demo_dropdown, threshold_slider],
        outputs=[scene_img, output_img, edge_img, results_text, mechanical_text]
    )

    # Đồng bộ ảnh sang các Tab khác khi chọn mẫu ở Tab 1
    demo_dropdown.change(
        fn=_load_demo_by_name,
        inputs=[demo_dropdown],
        outputs=[heatmap_input_img]
    )
    demo_dropdown.change(
        fn=_load_demo_by_name,
        inputs=[demo_dropdown],
        outputs=[texture_input_img]
    )
    demo_dropdown.change(
        fn=_load_demo_by_name,
        inputs=[demo_dropdown],
        outputs=[optical_input_img]
    )

    # Chuẩn hóa ảnh kéo thả trực tiếp vào gr.Image (sự kiện .upload)
    scene_img.upload(fn=_normalize_drag_drop, inputs=[scene_img], outputs=[scene_img])
    heatmap_input_img.upload(fn=_normalize_drag_drop, inputs=[heatmap_input_img], outputs=[heatmap_input_img])
    texture_input_img.upload(fn=_normalize_drag_drop, inputs=[texture_input_img], outputs=[texture_input_img])
    optical_input_img.upload(fn=_normalize_drag_drop, inputs=[optical_input_img], outputs=[optical_input_img])

    # === Tab 2 events ===
    upload_btn_t2.upload(
        fn=_normalize_upload,
        inputs=[upload_btn_t2],
        outputs=[heatmap_input_img]
    )
    demo_dropdown_t2.change(
        fn=_load_demo_by_name,
        inputs=[demo_dropdown_t2],
        outputs=[heatmap_input_img]
    )
    generate_heatmap_btn.click(
        fn=generate_dense_material_heatmap,
        inputs=[heatmap_input_img, patch_size_radio, alpha_slider, view_level_radio],
        outputs=[heatmap_output_img, heatmap_stats_text]
    )

    # === Tab 3 events ===
    upload_btn_t3.upload(
        fn=_normalize_upload,
        inputs=[upload_btn_t3],
        outputs=[texture_input_img]
    )
    demo_dropdown_t3.change(
        fn=_load_demo_by_name,
        inputs=[demo_dropdown_t3],
        outputs=[texture_input_img]
    )
    texture_analyze_btn.click(
        fn=run_texture_feature_analysis,
        inputs=[texture_input_img],
        outputs=[lbp_out_img, glcm_out_img, gabor_out_img, texture_stats_text]
    )

    # === Tab 4 events ===
    upload_btn_t4.upload(
        fn=_normalize_upload,
        inputs=[upload_btn_t4],
        outputs=[optical_input_img]
    )
    demo_dropdown_t4.change(
        fn=_load_demo_by_name,
        inputs=[demo_dropdown_t4],
        outputs=[optical_input_img]
    )
    optical_run_btn.click(
        fn=run_optical_simulation_tab,
        inputs=[optical_input_img, psf_slider, box_slider, bin_slider, photon_slider, noise_slider],
        outputs=[optical_result_img, optical_info_text, report_gallery_img]
    )

if __name__ == "__main__":
    print("-> [4/4] Đang khởi chạy máy chủ Gradio 4 Tabs...")
    print("=" * 65)
    print("🚀 GIAO DIỆN ĐÃ SẴN SÀNG! Mời bạn mở trình duyệt tại địa chỉ:")
    print("   👉 http://127.0.0.1:7860")
    print("=" * 65)
    sys.stdout.flush()
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
