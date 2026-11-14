import os
import sys
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def generate_pillbox_psf(radius=5):
    """Tạo kernel điểm tán xạ quang học dạng đĩa (Disk/Pillbox Optical PSF) mô phỏng nhòe tiêu cự ống kính"""
    size = 2 * radius + 1
    y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
    mask = x**2 + y**2 <= radius**2
    kernel = np.zeros((size, size), dtype=np.float32)
    kernel[mask] = 1.0 / np.sum(mask)
    return kernel

def apply_poisson_gaussian_noise(image_rgb, peak_photon=100.0, gaussian_sigma=12.0):
    """Mô phỏng nhiễu ảnh số: Nhiễu hạt photon Poisson + Nhiễu đọc cảm biến Gaussian"""
    img_float = image_rgb.astype(np.float32) / 255.0
    
    # 1. Poisson Shot Noise (Phụ thuộc vào cường độ thông lượng photon chiếu tới cảm biến)
    scaled = img_float * peak_photon
    poisson_noisy = np.random.poisson(np.clip(scaled, 0, None)) / peak_photon
    
    # 2. Additive Gaussian Read Noise (Nhiễu nhiệt vi mạch đọc cảm biến CMOS/CCD)
    gauss_noise = np.random.normal(0, gaussian_sigma / 255.0, img_float.shape)
    
    noisy_result = poisson_noisy + gauss_noise
    return np.clip(noisy_result * 255.0, 0, 255).astype(np.uint8)

def run_image_processing_pipeline(sample_img_path=None, output_path="outputs/image_processing_pipeline.png"):
    print("=== CHẠY QUY TRÌNH MÔ PHỎNG QUANG HỌC & XỬ LÝ ẢNH (ICCV 2025) ===")
    
    # 1. Lựa chọn ảnh thật Matador
    if sample_img_path is None or not os.path.exists(sample_img_path):
        candidates = [
            "matador/texture_img/multicap-2024-06-18_08-42-27-000000.tiff",
            "matador/texture_img/multicap-2024-06-18_08-52-49-000000.tiff",
            "matador/texture_img/multicap-2024-06-19_16-07-36-000000.tiff"
        ]
        for c in candidates:
            if os.path.exists(c):
                sample_img_path = c
                break
    
    if not sample_img_path or not os.path.exists(sample_img_path):
        raise FileNotFoundError(f"Không tìm thấy ảnh mẫu tại {sample_img_path}")
        
    print(f"-> Nạp ảnh vi kết cấu mẫu: {sample_img_path}")
    img_bgr = cv2.imread(sample_img_path)
    if img_bgr is None:
        raise ValueError("cv2.imread không đọc được ảnh.")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # Crop trung tâm 512x512
    h, w, _ = img_rgb.shape
    crop_size = min(512, h, w)
    start_y = (h - crop_size) // 2
    start_x = (w - crop_size) // 2
    orig_patch = img_rgb[start_y:start_y+crop_size, start_x:start_x+crop_size]
    
    # (1) Stage 1: Ảnh gốc nguyên bản (Original High-Res Patch)
    stage1 = orig_patch.copy()
    
    # (2) Stage 2: Phát hiện biên vi kết cấu (Sobel Gradient Magnitude & Canny Edges)
    gray = cv2.cvtColor(orig_patch, cv2.COLOR_RGB2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sobel_x**2 + sobel_y**2)
    sobel_norm = cv2.normalize(sobel_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    canny = cv2.Canny(gray, threshold1=50, threshold2=150)
    # Kết hợp trực quan hóa biên bằng bảng màu Infernal / Jet
    stage2 = cv2.applyColorMap(sobel_norm, cv2.COLORMAP_MAGMA)
    stage2 = cv2.cvtColor(stage2, cv2.COLOR_BGR2RGB)
    
    # (3) Stage 3: Lọc thông thấp nhòe tiêu cự (Defocus Blur via Disk/Gaussian PSF)
    disk_psf = generate_pillbox_psf(radius=7)
    stage3 = cv2.filter2D(orig_patch, -1, disk_psf)
    
    # (4) Stage 4: Lọc hộp (Box Filter) & Giảm mẫu Pixel Pitch (Pixel Binning Simulation)
    box_filtered = cv2.boxFilter(orig_patch, ddepth=-1, ksize=(8, 8))
    # Subsampling (Downsample 4x rồi phóng to Nearest Neighbor để thấy rõ kích thước ô pixel)
    down_small = cv2.resize(box_filtered, (crop_size // 4, crop_size // 4), interpolation=cv2.INTER_AREA)
    stage4 = cv2.resize(down_small, (crop_size, crop_size), interpolation=cv2.INTER_NEAREST)
    
    # (5) Stage 5: Mô phỏng nhiễu Poisson Shot Noise + Gaussian Read Noise
    stage5 = apply_poisson_gaussian_noise(orig_patch, peak_photon=80.0, gaussian_sigma=15.0)
    
    # Vẽ và xuất hình minh họa quy trình chuẩn ICCV 2025
    fig, axes = plt.subplots(1, 5, figsize=(22, 5.5))
    
    stages_data = [
        ("1. Ảnh gốc vi kết cấu\n(Original Patch 512×512)", stage1, None),
        ("2. Phát hiện biên vi kết cấu\n(Sobel Magnitude Map)", stage2, None),
        ("3. Lọc nhòe tiêu cự\n(Defocus Blur Pillbox PSF)", stage3, None),
        ("4. Lọc hộp & Giảm mẫu pixel\n(Box Filter & Pixel Pitch)", stage4, None),
        ("5. Cộng nhiễu cảm biến\n(Poisson + Gaussian Noise)", stage5, None)
    ]
    
    for ax, (title, img, cmap) in zip(axes, stages_data):
        ax.imshow(img, cmap=cmap)
        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
        ax.axis('off')
        
    plt.suptitle("Quy trình Mô phỏng Quang học & Xử lý Ảnh Học sâu (Matador Optical Simulation)", 
                 fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"-> Đã xuất đồ thị thành công vào: {output_path}")

if __name__ == "__main__":
    run_image_processing_pipeline()
