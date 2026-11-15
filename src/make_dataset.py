"""Sinh bộ ảnh cắt có nhãn cho bộ phân loại dấu (task.md 2B.1-2B.2): mô phỏng nhiều seed + mọi file thật có đáp án.

Mỗi lượt là một lần chạy scan_form.py --crops (ghi ảnh *_scan.png/json vào thư mục tạm, không đụng output/).
Cấu hình mô phỏng của mỗi seed (overshoot, xoay, ô lẫn cỡ, mực nhạt) rút ngẫu nhiên từ chính seed -> chạy lại ra y hệt.
Seed bắt đầu từ 100 để không trùng ma trận của regress.py.

Kết quả: data/crops/<doc>/labels.csv + ảnh (xem export_crops.py). Thư mục <doc> = "<mẫu>_sim<seed>" hoặc tên file thật.

Chạy:
    python src/make_dataset.py                       # 40 seed x 2 mẫu + file thật, ~10-15 phút
    python src/make_dataset.py --n 5 --only sim      # chạy thử nhanh
    python src/make_dataset.py --only real           # chỉ cắt lại file thật (sau khi thêm đáp án mới)
"""
import argparse
import csv
import os
import random
import shutil
import sys
import tempfile
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from common import CROPS, ROOT, SAMPLES, SIM_FORMS, TRUTH, run_scan

STYLES = "cross,tick,circle,fill,dash"


def sim_args(seed):
    r = random.Random(seed)
    a = ["--simulate", "--seed", str(seed), "--styles", STYLES,
         "--overshoot", str(r.choice([0.0, 0.2, 0.4, 0.6])),
         "--rotate", f"{r.uniform(-2.0, 2.0):.2f}",
         "--light", str(r.choice([0.0, 0.0, 0.3]))]
    return a + (["--mixed"] if r.random() < 0.5 else [])


def jobs(only, n, seed0):
    out = []
    if only in (None, "sim"):
        for f in SIM_FORMS:
            for s in range(seed0, seed0 + n):
                out.append({"key": f"sim {f[:3]} {s}", "pdf": SAMPLES / f"{f}.pdf", "args": sim_args(s)})
    if only in (None, "real"):
        for t in sorted(TRUTH.glob("*.json")):
            pdf = SAMPLES / f"{t.stem}.pdf"
            if pdf.exists():
                out.append({"key": f"real {t.stem}", "pdf": pdf, "args": []})
    return out


def run(job, tmp, crops):
    d = Path(tmp) / job["key"].replace(" ", "_")
    p = run_scan(job["pdf"], d, ["--crops", str(crops), *job["args"]])
    if p.returncode != 0:
        return "LỖI: " + (p.stderr.strip().splitlines()[-1] if p.stderr.strip() else f"mã lỗi {p.returncode}")
    return next((ln for ln in p.stdout.splitlines() if ln.startswith(("Ảnh cắt", "Không có đáp án"))), "?")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops", default=str(CROPS))
    ap.add_argument("--n", type=int, default=40, help="số seed mô phỏng cho mỗi mẫu")
    ap.add_argument("--seed0", type=int, default=100)
    ap.add_argument("--only", choices=["real", "sim"])
    ap.add_argument("-j", "--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    args = ap.parse_args()

    todo = jobs(args.only, args.n, args.seed0)
    tmp = tempfile.mkdtemp(prefix="dataset_")
    t0 = time.time()
    print(f"Chạy {len(todo)} lượt, {args.jobs} song song -> {args.crops}")
    with ThreadPoolExecutor(args.jobs) as ex:
        for job, msg in zip(todo, ex.map(lambda j: run(j, tmp, args.crops), todo)):
            print(f"  {job['key']}: {msg}")

    total = Counter()
    for f in Path(args.crops).glob("*/labels.csv"):
        with open(f, encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                total[(r["source"], r["kind"], r["label"])] += 1
    print("\nTổng trong", args.crops)
    for k, v in sorted(total.items()):
        print(f"  {k[0]:4} {k[1]:4} {k[2]:8} {v}")
    print(f"Thời gian: {time.time() - t0:.0f} giây")
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
