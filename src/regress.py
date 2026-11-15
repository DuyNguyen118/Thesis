"""Bộ kiểm thử hồi quy: chạy mọi file thật có đáp án + ma trận mô phỏng, so với regress_baseline.json.

Mỗi lần chạy scan_form.py ghi vào một thư mục tạm riêng (không đụng output/) nên chạy song song được.
Trả mã lỗi 1 nếu bất kỳ file/cấu hình nào kém hơn baseline:
  - file thật: tổng lỗi (sót ô + ô thừa + sót dấu + báo nhầm + hàng bảng sai) tăng, hoặc nhãn giảm > LABEL_DROP
  - mô phỏng: số dấu bị sót tăng, hoặc số báo nhầm tăng

Chạy:
    python src/regress.py
    python src/regress.py --update-baseline     # chỉ khi kết quả tốt hơn và đã kiểm tra bằng mắt
    python src/regress.py --only sim -j 2
"""
import argparse
import json
import os
import re
import shutil
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from common import BASELINE, SAMPLES, SIM_CONFIGS, SIM_FORMS, TRUTH, run_scan
from evaluate import evaluate, summary

LABEL_DROP = 2.0     # độ giống nhãn trung bình được phép giảm tối đa (dao động nhỏ do OCR)
SIM_RE = re.compile(r"nhận đúng (\d+)/(\d+) dấu đã vẽ, báo nhầm (\d+)")


def jobs(only):
    out = []
    if only in (None, "sim"):
        for f in SIM_FORMS:
            for a in SIM_CONFIGS:
                out.append({"key": f"sim {f[:3]} {' '.join(a)}", "pdf": SAMPLES / f"{f}.pdf",
                            "args": ["--simulate", *a]})
    if only in (None, "real"):
        for t in sorted(TRUTH.glob("*.json")):
            pdf = SAMPLES / f"{t.stem}.pdf"
            if pdf.exists():
                out.append({"key": f"real {t.stem}", "pdf": pdf, "args": [], "truth": t})
    return out


def run(job, tmp):
    d = Path(tmp) / re.sub(r"[^\w.-]+", "_", job["key"])
    p = run_scan(job["pdf"], d, job["args"])
    if p.returncode != 0:
        return {"crash": p.stderr.strip().splitlines()[-1] if p.stderr.strip() else f"mã lỗi {p.returncode}"}
    if "truth" in job:
        with open(d / f"{job['pdf'].stem}_scan.json", encoding="utf-8") as f:
            scan = json.load(f)
        with open(job["truth"], encoding="utf-8") as f:
            truth = json.load(f)
        r = evaluate(scan, truth)
        r["err"] = r["missed"] + r["extra"] + r["fn"] + r["fp"] + (r["rows"] - r["rows_ok"])
        return r
    m = SIM_RE.search(p.stdout)
    if not m:
        return {"crash": "không thấy dòng tổng kết mô phỏng"}
    ok, total, fp = map(int, m.groups())
    return {"ok": ok, "total": total, "missed": total - ok, "fp": fp,
            "errors": [ln.strip() for ln in p.stdout.splitlines() if "SAI:" in ln or "BÁO NHẦM" in ln]}


def worse(key, r, b):
    """Lý do kém hơn baseline (rỗng nếu không kém)."""
    if "crash" in r:
        return [f"lỗi chạy: {r['crash']}"]
    if b is None:
        return []
    if key.startswith("real"):
        why = [f"tổng lỗi {b['err']} -> {r['err']}"] if r["err"] > b["err"] else []
        if r["label"] is not None and b.get("label") is not None and r["label"] < b["label"] - LABEL_DROP:
            why.append(f"nhãn {b['label']} -> {r['label']}")
        return why
    why = []
    if r["missed"] > b["missed"]:
        why.append(f"sót dấu {b['missed']} -> {r['missed']}")
    if r["fp"] > b["fp"]:
        why.append(f"báo nhầm {b['fp']} -> {r['fp']}")
    return why


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--update-baseline", action="store_true", help="ghi kết quả lần này làm baseline mới")
    ap.add_argument("--only", choices=["real", "sim"])
    ap.add_argument("-j", "--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--keep", action="store_true", help="giữ thư mục tạm (ảnh, JSON) để xem lại")
    args = ap.parse_args()

    base = json.loads(BASELINE.read_text(encoding="utf-8")) if BASELINE.exists() else {}
    todo = jobs(args.only)
    tmp = tempfile.mkdtemp(prefix="regress_")
    t0 = time.time()
    print(f"Chạy {len(todo)} lượt, {args.jobs} song song, thư mục tạm {tmp}")
    with ThreadPoolExecutor(args.jobs) as ex:
        results = list(ex.map(lambda j: run(j, tmp), todo))

    failed = []
    sim_ok = sim_total = sim_fp = 0
    for job, r in zip(todo, results):
        key = job["key"]
        why = worse(key, r, base.get(key))
        mark = "KÉM HƠN" if why else ("MỚI" if key not in base else "ok")
        if "crash" in r:
            line = key
        elif key.startswith("real"):
            line = summary(key, r)
        else:
            line = f"{key}: đúng {r['ok']}/{r['total']}, báo nhầm {r['fp']}"
            sim_ok, sim_total, sim_fp = sim_ok + r["ok"], sim_total + r["total"], sim_fp + r["fp"]
        print(f"[{mark:7}] {line}" + (f"  <- {'; '.join(why)}" if why else ""))
        if why:
            failed.append(key)
            for e in r.get("errors", [])[:10]:
                print("            ", e)
    if sim_total:
        print(f"\nMô phỏng tổng: đúng {sim_ok}/{sim_total}, báo nhầm {sim_fp}")
    print(f"Thời gian: {time.time() - t0:.0f} giây")

    if args.update_baseline:
        if failed:
            print(f"\nKHÔNG cập nhật baseline: {len(failed)} lượt kém hơn.")
        else:
            keep = {"real": ("err", "label", "missed", "extra", "fn", "fp", "rows_ok", "rows"),
                    "sim": ("ok", "total", "missed", "fp")}
            for job, r in zip(todo, results):
                base[job["key"]] = {k: r[k] for k in keep[job["key"].split()[0]]}
            BASELINE.write_text(json.dumps(base, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"\nĐã cập nhật {BASELINE.name}")
    elif not base:
        print("\nChưa có baseline: chạy lại với --update-baseline để lưu mức hiện tại.")

    if args.keep:
        print(f"Giữ kết quả tại {tmp}")
    else:
        shutil.rmtree(tmp, ignore_errors=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
