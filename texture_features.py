"""
texture_features.py - Module trích xuất đặc trưng kết cấu cục bộ (Local Texture Descriptors)
thuộc phạm vi học phần Xử lý Ảnh, phục vụ cải thiện phân loại vật liệu ở tầng lá.

Ba bộ mô tả kết cấu kinh điển:
  1. LBP (Local Binary Pattern): Mô tả cấu trúc vi kết cấu lân cận (Ojala et al., 2002)
  2. GLCM (Gray-Level Co-occurrence Matrix): Thống kê tương quan mức xám (Haralick, 1973)
  3. Gabor Filters: Phân tích tần số hướng kết cấu (Manjunath & Ma, 1996)

Tích hợp vào pipeline: ảnh -> TextureFeatureExtractor -> vector 64-d -> nối với phi(x) 256-d -> 320-d
"""
import os
import sys
import numpy as np
import cv2
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def compute_lbp(gray, radius=1, n_points=8):
    """
    Tính Local Binary Pattern (LBP) cho ảnh xám.
    LBP mã hóa cấu trúc vi kết cấu cục bộ bằng cách so sánh mỗi pixel
    với các điểm lân cận trên đường tròn bán kính r.
    
    Args:
        gray: Ảnh xám uint8 (H, W)
        radius: Bán kính vòng tròn lân cận (mặc định 1)
        n_points: Số điểm lân cận trên vòng tròn (mặc định 8)
    Returns:
        lbp_image: Ma trận LBP (H, W), giá trị 0..2^n_points - 1
    """
    h, w = gray.shape
    lbp = np.zeros((h, w), dtype=np.uint32)
    
    for i in range(n_points):
        angle = 2.0 * np.pi * i / n_points
        dx = radius * np.cos(angle)
        dy = -radius * np.sin(angle)
        
        # Nội suy song tuyến (Bilinear Interpolation) cho vị trí sub-pixel
        x0 = int(np.floor(dx))
        y0 = int(np.floor(dy))
        x1 = x0 + 1
        y1 = y0 + 1
        
        fx = dx - x0
        fy = dy - y0
        
        pad_len = int(np.ceil(radius)) + 1
        padded = np.pad(gray, pad_len, mode='reflect').astype(np.float32)
        
        # 4 điểm lân cận trên lưới pixel
        r = pad_len
        tl = padded[r + y0:r + y0 + h, r + x0:r + x0 + w]
        tr = padded[r + y0:r + y0 + h, r + x1:r + x1 + w]
        bl = padded[r + y1:r + y1 + h, r + x0:r + x0 + w]
        br = padded[r + y1:r + y1 + h, r + x1:r + x1 + w]
        
        # Nội suy song tuyến
        neighbor = (1 - fx) * (1 - fy) * tl + fx * (1 - fy) * tr + \
                   (1 - fx) * fy * bl + fx * fy * br
        
        # So sánh ngưỡng: nếu neighbor >= center thì bit = 1
        center = gray.astype(np.float32)
        bit = (neighbor >= center).astype(np.uint32)
        lbp |= (bit << i)
    
    return lbp.astype(np.uint8) if n_points <= 8 else lbp


def compute_lbp_histogram(gray, radius=1, n_points=8, n_bins=32):
    """
    Tính histogram của LBP, chuẩn hóa L1 -> vector đặc trưng.
    
    Args:
        gray: Ảnh xám uint8 (H, W)
        n_bins: Số bin histogram (mặc định 32)
    Returns:
        hist: Vector histogram LBP chuẩn hóa (n_bins,)
    """
    lbp = compute_lbp(gray, radius, n_points)
    max_val = 2 ** n_points
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, max_val))
    # Chuẩn hóa L1
    hist = hist.astype(np.float32)
    total = hist.sum()
    if total > 0:
        hist /= total
    return hist


def compute_glcm_features(gray, distances=[1, 3], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
    """
    Tính các đặc trưng Haralick từ Gray-Level Co-occurrence Matrix (GLCM).
    
    4 đặc trưng chính:
      - Contrast: Đo sự chênh lệch cường độ giữa pixel và lân cận
      - Dissimilarity: Tương tự contrast nhưng tuyến tính
      - Homogeneity (IDM): Đo tính đồng nhất cục bộ
      - Energy (ASM): Đo tính đều đặn của kết cấu
    
    Args:
        gray: Ảnh xám uint8 (H, W)
        distances: Danh sách khoảng cách (pixel)
        angles: Danh sách góc (radian)
    Returns:
        features: Vector đặc trưng GLCM (num_distances * num_angles * 4,)
    """
    # Lượng tử hóa xuống 16 mức xám để giảm chi phí tính toán
    n_levels = 16
    quantized = (gray.astype(np.float32) / 256.0 * n_levels).astype(np.int32)
    quantized = np.clip(quantized, 0, n_levels - 1)
    
    features = []
    h, w = quantized.shape
    
    for d in distances:
        for angle in angles:
            # Tính vector dịch chuyển (offset)
            dy = -int(round(d * np.sin(angle)))
            dx = int(round(d * np.cos(angle)))
            
            # Khởi tạo ma trận đồng hiện
            glcm = np.zeros((n_levels, n_levels), dtype=np.float64)
            
            # Tính GLCM
            y_start = max(0, -dy)
            y_end = min(h, h - dy)
            x_start = max(0, -dx)
            x_end = min(w, w - dx)
            
            for y in range(y_start, y_end):
                for x in range(x_start, x_end):
                    i_val = quantized[y, x]
                    j_val = quantized[y + dy, x + dx]
                    glcm[i_val, j_val] += 1
            
            # Đối xứng hóa và chuẩn hóa
            glcm = glcm + glcm.T
            total = glcm.sum()
            if total > 0:
                glcm /= total
            
            # Trích xuất 4 đặc trưng Haralick
            i_idx, j_idx = np.meshgrid(range(n_levels), range(n_levels), indexing='ij')
            i_idx = i_idx.astype(np.float64)
            j_idx = j_idx.astype(np.float64)
            
            contrast = np.sum(glcm * (i_idx - j_idx) ** 2)
            dissimilarity = np.sum(glcm * np.abs(i_idx - j_idx))
            homogeneity = np.sum(glcm / (1.0 + (i_idx - j_idx) ** 2))
            energy = np.sum(glcm ** 2)
            
            features.extend([contrast, dissimilarity, homogeneity, energy])
    
    return np.array(features, dtype=np.float32)


def compute_gabor_features(gray, ksize=31, num_orientations=4, num_scales=3):
    """
    Trích xuất đặc trưng bộ lọc Gabor đa hướng, đa tần số.
    
    Gabor filter g(x,y; λ,θ,σ,γ) phân tích năng lượng kết cấu theo
    từng hướng θ và tần số không gian 1/λ.
    
    Args:
        gray: Ảnh xám uint8 (H, W)
        ksize: Kích thước kernel (pixel)
        num_orientations: Số hướng θ (mặc định 4: 0°, 45°, 90°, 135°)
        num_scales: Số tỉ lệ tần số λ (mặc định 3)
    Returns:
        features: Vector đặc trưng Gabor (num_orientations * num_scales * 2,)
            Mỗi cặp (mean, std) của phản hồi Gabor.
    """
    features = []
    gray_f = gray.astype(np.float32) / 255.0
    
    for s in range(num_scales):
        # Bước sóng λ tăng dần: λ = {4, 8, 16} pixel
        lambd = 4.0 * (2 ** s)
        sigma = lambd * 0.56  # Băng thông cố định
        gamma = 0.5  # Tỷ lệ khung hình (Aspect Ratio) của nhân Gabor
        
        for o in range(num_orientations):
            theta = o * np.pi / num_orientations
            
            # Tạo kernel Gabor
            kernel = cv2.getGaborKernel(
                (ksize, ksize), sigma, theta, lambd, gamma, psi=0, ktype=cv2.CV_32F
            )
            kernel /= (1.5 * kernel.sum() + 1e-7)
            
            # Lọc ảnh bằng kernel Gabor
            filtered = cv2.filter2D(gray_f, cv2.CV_32F, kernel)
            
            # Trích xuất trung bình và độ lệch chuẩn năng lượng
            mean_energy = np.abs(filtered).mean()
            std_energy = np.abs(filtered).std()
            features.extend([mean_energy, std_energy])
    
    return np.array(features, dtype=np.float32)


def extract_texture_features(image_rgb, target_dim=64):
    """
    Hàm chính: Trích xuất toàn bộ đặc trưng kết cấu từ ảnh RGB.
    Kết hợp LBP + GLCM + Gabor -> vector chuẩn hóa target_dim chiều.
    
    Args:
        image_rgb: Ảnh RGB uint8 (H, W, 3) hoặc đường dẫn file
        target_dim: Số chiều vector đầu ra (mặc định 64)
    Returns:
        feature_vector: numpy array (target_dim,) float32
    """
    # Xử lý đầu vào
    if isinstance(image_rgb, str):
        img = cv2.imread(image_rgb)
        if img is None:
            return np.zeros(target_dim, dtype=np.float32)
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    if isinstance(image_rgb, Image.Image):
        image_rgb = np.array(image_rgb.convert('RGB'))
    
    # Chuyển sang ảnh xám
    if image_rgb.ndim == 3:
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_rgb
    
    # Resize về 128x128 để tốc độ tính toán ổn định
    gray_resized = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA)
    
    # 1. LBP histogram (32-d)
    lbp_hist = compute_lbp_histogram(gray_resized, radius=1, n_points=8, n_bins=32)
    
    # 2. GLCM features: 2 khoảng cách × 4 hướng × 4 đặc trưng = 32-d
    glcm_feat = compute_glcm_features(gray_resized, distances=[1, 3])
    
    # 3. Gabor features: 3 tần số × 4 hướng × 2 (mean, std) = 24-d
    gabor_feat = compute_gabor_features(gray_resized, num_orientations=4, num_scales=3)
    
    # Nối tất cả: 32 + 32 + 24 = 88-d
    raw_features = np.concatenate([lbp_hist, glcm_feat, gabor_feat])
    
    # Chuẩn hóa L2
    norm = np.linalg.norm(raw_features)
    if norm > 0:
        raw_features = raw_features / norm
    
    # Chiếu về target_dim bằng cắt/pad (hoặc PCA nếu có sklearn)
    if len(raw_features) >= target_dim:
        feature_vector = raw_features[:target_dim]
    else:
        feature_vector = np.pad(raw_features, (0, target_dim - len(raw_features)))
    
    return feature_vector.astype(np.float32)


def visualize_texture_analysis(image_rgb, output_path=None):
    """
    Tạo bảng trực quan hóa phân tích kết cấu: LBP, GLCM Energy Map, Gabor Responses.
    
    Args:
        image_rgb: Ảnh RGB uint8 (H, W, 3)
        output_path: Đường dẫn lưu ảnh kết quả (None = không lưu, trả về figure)
    Returns:
        fig: Matplotlib figure object
    """
    import matplotlib.pyplot as plt
    
    if image_rgb.ndim == 3:
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_rgb
    
    gray_128 = cv2.resize(gray, (128, 128))
    
    # 1. LBP image
    lbp_img = compute_lbp(gray_128, radius=1, n_points=8)
    
    # 2. GLCM Energy Map (sliding window)
    win_size = 16
    h, w = gray_128.shape
    energy_map = np.zeros((h, w), dtype=np.float32)
    for y in range(0, h - win_size, win_size // 2):
        for x in range(0, w - win_size, win_size // 2):
            patch = gray_128[y:y+win_size, x:x+win_size]
            glcm_f = compute_glcm_features(patch, distances=[1], angles=[0])
            energy_map[y:y+win_size, x:x+win_size] = glcm_f[3]  # energy value
    
    # 3. Gabor responses (4 hướng)
    gabor_responses = []
    gray_f = gray_128.astype(np.float32) / 255.0
    for o in range(4):
        theta = o * np.pi / 4
        kernel = cv2.getGaborKernel((31, 31), 4.0*0.56, theta, 4.0, 0.5, 0, cv2.CV_32F)
        kernel /= (1.5 * kernel.sum() + 1e-7)
        resp = cv2.filter2D(gray_f, cv2.CV_32F, kernel)
        gabor_responses.append(np.abs(resp))
    
    # 4. Sobel Magnitude
    sx = cv2.Sobel(gray_128, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray_128, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sx**2 + sy**2)
    
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    
    axes[0, 0].imshow(image_rgb if image_rgb.ndim == 3 else gray, cmap='gray' if image_rgb.ndim == 2 else None)
    axes[0, 0].set_title("Ảnh gốc (Original)", fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(lbp_img, cmap='gray')
    axes[0, 1].set_title("LBP (Local Binary Pattern)", fontweight='bold')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(energy_map, cmap='hot')
    axes[0, 2].set_title("GLCM Energy Map", fontweight='bold')
    axes[0, 2].axis('off')
    
    axes[0, 3].imshow(sobel_mag, cmap='magma')
    axes[0, 3].set_title("Sobel Gradient Magnitude", fontweight='bold')
    axes[0, 3].axis('off')
    
    gabor_titles = ["Gabor 0°", "Gabor 45°", "Gabor 90°", "Gabor 135°"]
    for i in range(4):
        axes[1, i].imshow(gabor_responses[i], cmap='inferno')
        axes[1, i].set_title(gabor_titles[i], fontweight='bold')
        axes[1, i].axis('off')
    
    plt.suptitle("Phân tích Đặc trưng Kết cấu Cục bộ (Texture Feature Analysis)",
                 fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        print(f"-> Đã lưu trực quan hóa kết cấu vào: {output_path}")
    
    return fig


if __name__ == "__main__":
    import os
    import glob
    
    # Test trích xuất đặc trưng trên ảnh demo
    demo_files = sorted(glob.glob("demo_samples/*.png"))
    if demo_files:
        print("=== KIỂM THỬ TRÍCH XUẤT ĐẶC TRƯNG KẾT CẤU ===")
        for f in demo_files[:3]:
            img = cv2.imread(f)
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                feat = extract_texture_features(img_rgb, target_dim=64)
                name = os.path.basename(f)
                print(f"  {name}: shape={feat.shape}, min={feat.min():.4f}, max={feat.max():.4f}, norm={np.linalg.norm(feat):.4f}")
        
        # Tạo trực quan hóa cho ảnh đầu tiên
        first_img = cv2.cvtColor(cv2.imread(demo_files[0]), cv2.COLOR_BGR2RGB)
        visualize_texture_analysis(first_img, output_path="outputs/texture_analysis_demo.png")
    else:
        print("Không tìm thấy ảnh demo. Chạy từ thư mục gốc dự án.")
