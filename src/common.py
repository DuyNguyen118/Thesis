"""Shared paths, constants, and execution utilities for the thesis project."""
import os
import subprocess
import sys
from pathlib import Path

# Thư mục gốc và các thư mục dữ liệu chuẩn
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
SAMPLES = DATA / "samples"
TRUTH = DATA / "truth"
CROPS = DATA / "crops"
MODELS = ROOT / "models"
OUTPUT = ROOT / "output"
BASELINE = ROOT / "regress_baseline.json"

# Danh sách mẫu form và cấu hình mô phỏng chuẩn
SIM_FORMS = ["M02_Intern-evaluation-template", "M03_LABOUR-MARKET-FEEDBACK"]
SIM_CONFIGS = [
    ["--seed", "1"],
    ["--seed", "7", "--overshoot", "0.3", "--rotate", "1.5"],
    ["--seed", "3", "--mixed"],
    ["--seed", "5", "--overshoot", "0.5", "--rotate", "-1.0"],
    ["--seed", "11", "--overshoot", "0.4", "--rotate", "2.0", "--mixed"],
    ["--seed", "21", "--overshoot", "0.5", "--rotate", "-1.8"],
    ["--seed", "33", "--mixed", "--rotate", "0.8"],
    ["--seed", "44", "--overshoot", "0.3", "--mixed"],
    ["--seed", "55", "--overshoot", "0.2", "--rotate", "1.0", "--mixed"],
]


def run_scan(pdf_path, out_dir, extra_args=None):
    """Chạy scan_form.py an toàn với UTF-8 và Tesseract đơn luồng."""
    cmd = [
        sys.executable,
        str(ROOT / "src" / "scan_form.py"),
        str(pdf_path),
        "--out",
        str(out_dir),
    ]
    if extra_args:
        cmd.extend(extra_args)
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "OMP_THREAD_LIMIT": "1"}
    return subprocess.run(
        cmd,
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

