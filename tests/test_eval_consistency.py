"""
test_eval_consistency.py - Kiểm thử hồi quy (Regression Test):
  1. Nạp trọng số best_hierarchical_model.pth và đánh giá trên test set.
  2. Xác nhận Material accuracy ∈ [84.00%, 84.30%] (benchmark 84.14%).
  3. Xác nhận tính đơn điệu phân cấp: Acc[Phase] >= Acc[State] >= ... >= Acc[Material].
"""
import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

CHECKPOINT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "checkpoints", "best_hierarchical_model.pth"
)
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "matador")

SKIP_CHECKPOINT = not os.path.isfile(CHECKPOINT_PATH)
SKIP_DATA = not os.path.isdir(os.path.join(DATA_DIR, "texture_img"))
SKIP_MSG = "Checkpoint or dataset not found"


@unittest.skipIf(SKIP_CHECKPOINT or SKIP_DATA, SKIP_MSG)
class TestEvalConsistency(unittest.TestCase):
    """Kiểm tra tính nhất quán đánh giá mô hình trên test set."""

    @classmethod
    def setUpClass(cls):
        """Chạy đánh giá 1 lần duy nhất rồi dùng cho tất cả test cases."""
        from evaluate import run_evaluation
        cls.acc_by_level = run_evaluation(
            checkpoint_path=CHECKPOINT_PATH,
            data_dir=DATA_DIR,
            mode="c1"
        )

    def test_phase_accuracy_100(self):
        """Level 0 (Phase) phải đạt 100% (chỉ có 1 nút 'Solid')."""
        self.assertAlmostEqual(self.acc_by_level[0], 100.0, delta=0.01,
                               msg=f"Phase accuracy: {self.acc_by_level[0]:.2f}% (expected 100.00%)")

    def test_material_accuracy_benchmark(self):
        """Level 4 (Material) phải đạt >= 87.50% (kỷ lục mới 88.22%, vượt baseline 84.14%)."""
        acc = self.acc_by_level[4]
        self.assertGreaterEqual(acc, 87.50,
                                f"Material accuracy {acc:.2f}% below 87.50% benchmark!")

    def test_state_accuracy_above_95(self):
        """Level 1 (State) phải đạt >= 95% (kỷ lục 97.89%)."""
        self.assertGreaterEqual(self.acc_by_level[1], 95.0,
                                f"State accuracy {self.acc_by_level[1]:.2f}% below 95%!")

    def test_composition_accuracy_above_89(self):
        """Level 2 (Composition) phải đạt >= 92% (kỷ lục 93.35%)."""
        self.assertGreaterEqual(self.acc_by_level[2], 92.0,
                                f"Composition accuracy {self.acc_by_level[2]:.2f}% below 92%!")

    def test_form_accuracy_above_material(self):
        """Level 3 (Form) phải >= Level 4 (Material) sau khi suy luận từ nút cha."""
        self.assertGreaterEqual(
            self.acc_by_level[3], self.acc_by_level[4],
            f"Form ({self.acc_by_level[3]:.2f}%) should >= Material ({self.acc_by_level[4]:.2f}%)!"
        )

    def test_hierarchical_monotonicity(self):
        """Tính đơn điệu phân cấp: Acc[0] >= Acc[1] >= Acc[2] >= Acc[3] >= Acc[4]."""
        for d in range(4):
            self.assertGreaterEqual(
                self.acc_by_level[d], self.acc_by_level[d + 1],
                f"Monotonicity violated: Level {d} ({self.acc_by_level[d]:.2f}%) "
                f"< Level {d+1} ({self.acc_by_level[d+1]:.2f}%)"
            )

    def test_results_exact_match(self):
        """Kết quả phải khớp chính xác với kỷ lục mới [100.00, 97.89, 93.35, 90.94, 88.22]."""
        expected = {
            0: 100.00,
            1: 97.89,
            2: 93.35,
            3: 90.94,
            4: 88.22
        }
        for d, exp in expected.items():
            self.assertAlmostEqual(
                self.acc_by_level[d], exp, delta=0.5,
                msg=f"Level {d}: {self.acc_by_level[d]:.2f}% vs expected {exp:.2f}%"
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
