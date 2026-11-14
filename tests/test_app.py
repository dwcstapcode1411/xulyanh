"""
test_app.py - Kiểm thử các hàm hỗ trợ trong app.py: load_image_as_rgb_uint8, chuyển đổi TIFF.
"""
import sys
import os
import unittest
import glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PIL import Image

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEMO_DIR = os.path.join(PROJECT_DIR, "demo_samples")
DATA_DIR = os.path.join(PROJECT_DIR, "matador")


def load_image_as_rgb_uint8(file_path):
    """
    Hàm load ảnh an toàn (copy từ app.py) hỗ trợ cả TIFF 8-bit và 16-bit.
    Trả về numpy array (H, W, 3) uint8.
    """
    img = Image.open(file_path)

    # Chuyển 16-bit sang 8-bit
    arr = np.array(img)
    if arr.dtype == np.uint16:
        arr = (arr / 256).astype(np.uint8)

    # Chuyển về RGB
    img = Image.fromarray(arr)
    if img.mode != 'RGB':
        img = img.convert('RGB')

    return np.array(img)


class TestLoadImageAsRGBUint8(unittest.TestCase):
    """Kiểm tra hàm load_image_as_rgb_uint8."""

    @unittest.skipIf(not os.path.isdir(DEMO_DIR), "Demo samples not found")
    def test_load_png_demo_sample(self):
        """Đọc file PNG từ demo_samples phải trả về array (H, W, 3) uint8."""
        pngs = glob.glob(os.path.join(DEMO_DIR, "*.png"))
        if not pngs:
            self.skipTest("No PNG files in demo_samples/")

        img = load_image_as_rgb_uint8(pngs[0])
        self.assertEqual(len(img.shape), 3)
        self.assertEqual(img.shape[2], 3)
        self.assertEqual(img.dtype, np.uint8)

    @unittest.skipIf(not os.path.isdir(os.path.join(DATA_DIR, "texture_img")), "Matador texture_img not found")
    def test_load_tiff_file(self):
        """Đọc file TIFF từ matador/texture_img phải trả về array (H, W, 3) uint8."""
        tiffs = glob.glob(os.path.join(DATA_DIR, "texture_img", "*.tiff"))
        if not tiffs:
            self.skipTest("No TIFF files found")

        img = load_image_as_rgb_uint8(tiffs[0])
        self.assertEqual(len(img.shape), 3)
        self.assertEqual(img.shape[2], 3)
        self.assertEqual(img.dtype, np.uint8)
        self.assertEqual(img.shape[0], 512)
        self.assertEqual(img.shape[1], 512)

    @unittest.skipIf(not os.path.isdir(os.path.join(DATA_DIR, "texture_img")), "Matador texture_img not found")
    def test_pixel_range_valid(self):
        """Giá trị pixel phải nằm trong [0, 255]."""
        tiffs = glob.glob(os.path.join(DATA_DIR, "texture_img", "*.tiff"))
        if not tiffs:
            self.skipTest("No TIFF files found")

        img = load_image_as_rgb_uint8(tiffs[0])
        self.assertTrue(img.min() >= 0)
        self.assertTrue(img.max() <= 255)

    def test_load_synthetic_uint16_image(self):
        """Tạo ảnh giả uint16 và verify chuyển đổi sang uint8."""
        import tempfile
        # Tạo ảnh uint16 với giá trị đặc trưng
        arr_16 = np.random.randint(0, 65535, (100, 100, 3), dtype=np.uint16)
        img_16 = Image.fromarray(arr_16)

        with tempfile.NamedTemporaryFile(suffix='.tiff', delete=False) as f:
            img_16.save(f.name)
            result = load_image_as_rgb_uint8(f.name)
            os.unlink(f.name)

        self.assertEqual(result.dtype, np.uint8)
        self.assertEqual(result.shape, (100, 100, 3))


class TestDemoSamplesIntegrity(unittest.TestCase):
    """Kiểm tra tính toàn vẹn của demo samples."""

    @unittest.skipIf(not os.path.isdir(DEMO_DIR), "Demo samples not found")
    def test_10_demo_samples_exist(self):
        """Phải có 10 file PNG demo samples."""
        pngs = glob.glob(os.path.join(DEMO_DIR, "*.png"))
        self.assertEqual(len(pngs), 10, f"Expected 10 demo PNGs, found {len(pngs)}")

    @unittest.skipIf(not os.path.isdir(DEMO_DIR), "Demo samples not found")
    def test_demo_samples_valid_images(self):
        """Mọi demo sample phải là ảnh hợp lệ có thể mở được."""
        pngs = glob.glob(os.path.join(DEMO_DIR, "*.png"))
        for png_path in pngs:
            img = Image.open(png_path)
            self.assertGreater(img.width, 0)
            self.assertGreater(img.height, 0)


class TestOutputsExist(unittest.TestCase):
    """Kiểm tra sự tồn tại của các biểu đồ đầu ra."""

    def test_figure_5b_exists(self):
        """Biểu đồ figure_5b_hierarchy_accuracy.png phải tồn tại."""
        path = os.path.join(PROJECT_DIR, "outputs", "figure_5b_hierarchy_accuracy.png")
        self.assertTrue(os.path.isfile(path), f"Missing: {path}")

    def test_confusion_matrix_exists(self):
        """Biểu đồ confusion_matrix.png phải tồn tại."""
        path = os.path.join(PROJECT_DIR, "outputs", "confusion_matrix.png")
        self.assertTrue(os.path.isfile(path), f"Missing: {path}")

    def test_ablation_comparison_exists(self):
        """Biểu đồ ablation_comparison.png phải tồn tại."""
        path = os.path.join(PROJECT_DIR, "outputs", "ablation_comparison.png")
        self.assertTrue(os.path.isfile(path), f"Missing: {path}")

    def test_optical_simulation_exists(self):
        """Biểu đồ image_processing_pipeline.png phải tồn tại."""
        path = os.path.join(PROJECT_DIR, "outputs", "image_processing_pipeline.png")
        self.assertTrue(os.path.isfile(path), f"Missing: {path}")


class TestCheckpointIntegrity(unittest.TestCase):
    """Kiểm tra tính toàn vẹn của file checkpoint."""

    @unittest.skipIf(not os.path.isfile(os.path.join(PROJECT_DIR, "checkpoints", "best_hierarchical_model.pth")),
                     "Checkpoint not found")
    def test_checkpoint_loadable(self):
        """Checkpoint phải nạp được thành công."""
        ckpt_path = os.path.join(PROJECT_DIR, "checkpoints", "best_hierarchical_model.pth")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        self.assertIn("feature_extractor_state", ckpt)
        self.assertIn("gat_model_state", ckpt)

    @unittest.skipIf(not os.path.isfile(os.path.join(PROJECT_DIR, "checkpoints", "backup_84_14_model.pth")),
                     "Backup checkpoint not found")
    def test_backup_checkpoint_exists_and_loadable(self):
        """Backup checkpoint phải tồn tại và nạp được."""
        ckpt_path = os.path.join(PROJECT_DIR, "checkpoints", "backup_84_14_model.pth")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        self.assertIn("feature_extractor_state", ckpt)
        self.assertIn("gat_model_state", ckpt)

    @unittest.skipIf(
        not os.path.isfile(os.path.join(PROJECT_DIR, "checkpoints", "best_hierarchical_model.pth")) or
        not os.path.isfile(os.path.join(PROJECT_DIR, "checkpoints", "backup_84_14_model.pth")),
        "Checkpoints not found"
    )
    def test_best_equals_backup(self):
        """best_hierarchical_model.pth phải giống hệt backup_84_14_model.pth (cùng kích thước file)."""
        best_path = os.path.join(PROJECT_DIR, "checkpoints", "best_hierarchical_model.pth")
        backup_path = os.path.join(PROJECT_DIR, "checkpoints", "backup_84_14_model.pth")
        best_size = os.path.getsize(best_path)
        backup_size = os.path.getsize(backup_path)
        self.assertEqual(best_size, backup_size,
                         f"best ({best_size}) != backup ({backup_size}) bytes!")


class TestNewAppFeatures(unittest.TestCase):
    """Kiểm tra các tính năng nâng cao trong app.py (Heatmap, Texture Analysis, Optical Simulation)."""

    def setUp(self):
        # Tạo ảnh mẫu RGB uint8 128x128
        np.random.seed(42)
        self.sample_img = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)

    def test_dense_material_heatmap_generation(self):
        """Hàm generate_dense_material_heatmap phải tạo được overlay image và bảng thống kê."""
        from app import generate_dense_material_heatmap
        overlay, stats = generate_dense_material_heatmap(self.sample_img, grid_res=4, blend_alpha=0.45)
        self.assertIsNotNone(overlay)
        self.assertEqual(overlay.shape, self.sample_img.shape)
        self.assertEqual(overlay.dtype, np.uint8)
        self.assertIn("Composition", stats)

    def test_run_texture_feature_analysis(self):
        """Hàm run_texture_feature_analysis phải tạo được ảnh kết quả và thống kê vector."""
        from app import run_texture_feature_analysis
        res_img, stats = run_texture_feature_analysis(self.sample_img)
        self.assertIsNotNone(res_img)
        self.assertIn("64-D", stats)

    def test_run_optical_simulation_tab(self):
        """Hàm run_optical_simulation_tab phải mô phỏng được 5 giai đoạn quang học."""
        from app import run_optical_simulation_tab
        res_img, info = run_optical_simulation_tab(self.sample_img)
        self.assertIsNotNone(res_img)
        self.assertIn("Giai đoạn", info)


if __name__ == "__main__":
    unittest.main(verbosity=2)
