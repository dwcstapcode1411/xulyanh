"""
run_tests.py - Điểm khởi chạy duy nhất cho toàn bộ bộ kiểm thử tự động Matador Project.
Chạy: python run_tests.py
       python run_tests.py --fast        (bỏ qua test_eval_consistency cần GPU)
       python run_tests.py --verbose     (hiển thị chi tiết)
       python run_tests.py -k test_name  (chạy một test cụ thể)
"""
import sys
import os
import warnings
import unittest
import argparse
import time

# Tắt cảnh báo rác
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="Matador Project - Automated Test Suite Runner")
    parser.add_argument("--fast", action="store_true",
                        help="Skip slow tests (eval consistency with GPU inference)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose output")
    parser.add_argument("-k", "--keyword", type=str, default=None,
                        help="Only run tests matching keyword")
    args = parser.parse_args()

    # Thêm thư mục gốc vào PATH
    project_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, project_dir)

    print("=" * 70)
    print("  MATADOR PROJECT - BỘ KIỂM THỬ TỰ ĐỘNG")
    print("  Hierarchical Material Classification (ICCV 2025)")
    print("=" * 70)
    print(f"  Thời gian: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Thư mục dự án: {project_dir}")
    print(f"  Chế độ: {'FAST (bỏ qua eval GPU)' if args.fast else 'FULL'}")
    print("=" * 70)

    # Tải test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    test_modules = [
        "tests.test_taxonomy",
        "tests.test_dataset",
        "tests.test_models",
        "tests.test_app",
        "tests.test_improvements",
    ]

    if not args.fast:
        test_modules.append("tests.test_eval_consistency")

    for module_name in test_modules:
        try:
            module_suite = loader.loadTestsFromName(module_name)
            if args.keyword:
                filtered = unittest.TestSuite()
                for test_group in module_suite:
                    for test_case in test_group:
                        if args.keyword.lower() in str(test_case).lower():
                            filtered.addTest(test_case)
                suite.addTest(filtered)
            else:
                suite.addTest(module_suite)
            print(f"  [OK] Loaded: {module_name}")
        except Exception as e:
            print(f"  [ERROR] Failed to load {module_name}: {e}")

    print("=" * 70)
    print()

    # Chạy test
    verbosity = 2 if args.verbose else 1
    runner = unittest.TextTestRunner(verbosity=verbosity, stream=sys.stdout)
    start_time = time.time()
    result = runner.run(suite)
    elapsed = time.time() - start_time

    # Tổng kết
    print()
    print("=" * 70)
    print("  TỔNG KẾT KẾT QUẢ KIỂM THỬ")
    print("=" * 70)
    total = result.testsRun
    passed = total - len(result.failures) - len(result.errors) - len(result.skipped)
    print(f"  Tổng test    : {total}")
    print(f"  Passed       : {passed}")
    print(f"  Failed       : {len(result.failures)}")
    print(f"  Errors       : {len(result.errors)}")
    print(f"  Skipped      : {len(result.skipped)}")
    print(f"  Thời gian    : {elapsed:.1f}s")
    print(f"  Kết quả      : {'PASSED ✅' if result.wasSuccessful() else 'FAILED ❌'}")
    print("=" * 70)

    if result.failures:
        print("\n  CHI TIẾT FAILURES:")
        for test, traceback in result.failures:
            print(f"\n  ❌ {test}:")
            for line in traceback.strip().split('\n'):
                print(f"     {line}")

    if result.errors:
        print("\n  CHI TIẾT ERRORS:")
        for test, traceback in result.errors:
            print(f"\n  ⚠️ {test}:")
            for line in traceback.strip().split('\n'):
                print(f"     {line}")

    sys.exit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
