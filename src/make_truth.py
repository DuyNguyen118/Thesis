"""Sinh NHÁP đáp án từ kết quả quét để người sửa tay -> data/truth/<stem>.json.

Máy đã tìm vị trí ô và đoán có dấu hay không; người chỉ cần mở ảnh output/<stem>_p*_scan.png,
sửa chỗ máy sai (marked, nhãn, ô thiếu/thừa, cột bảng) rồi đặt "verified": true.
evaluate.py vẫn chấm file chưa verified nhưng in cảnh báo.

Chạy:
    python src/make_truth.py output/Data1.1_scan.json
    python src/make_truth.py output/*_scan.json --force      # ghi đè đáp án đã có
"""
import argparse
import json
from pathlib import Path

from common import TRUTH


def draft(scan):
    pages = []
    for pno, p in enumerate(scan, 1):
        boxes = [{"x": b["x"], "y": b["y"], "w": b["w"], "h": b["h"],
                  "label": b.get("text", ""), "marked": bool(b["marked"])} for b in p["boxes"]]
        tables = [{"index": t["index"], "bbox": [int(v) for v in t["bbox"]], "header": t["header"],
                   "rows": [{"row": r["row"], "label": r["label"], "marked_cols": r["marked_cols"]}
                            for r in t["rows"]]}
                  for t in p["tables"]]
        pages.append({"page": pno, "boxes": boxes, "tables": tables})
    return pages


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scans", nargs="+", help="output/<stem>_scan.json")
    ap.add_argument("--truth-dir", default=str(TRUTH))
    ap.add_argument("--force", action="store_true", help="ghi đè đáp án đã có")
    args = ap.parse_args()

    tdir = Path(args.truth_dir)
    tdir.mkdir(parents=True, exist_ok=True)
    for s in args.scans:
        stem = Path(s).name.removesuffix("_scan.json")
        dst = tdir / f"{stem}.json"
        if dst.exists() and not args.force:
            print(f"BỎ QUA {dst}: đã có (dùng --force để ghi đè)")
            continue
        with open(s, encoding="utf-8") as f:
            scan = json.load(f)
        truth = {"source": stem, "verified": False,
                 "note": "Nháp sinh từ kết quả máy. Kiểm tra bằng mắt rồi đặt verified = true.",
                 "pages": draft(scan)}
        with open(dst, "w", encoding="utf-8") as f:
            json.dump(truth, f, ensure_ascii=False, indent=2)
        n_box = sum(len(p["boxes"]) for p in truth["pages"])
        n_row = sum(len(t["rows"]) for p in truth["pages"] for t in p["tables"])
        print(f"Đã ghi {dst}: {n_box} ô, {n_row} hàng bảng")


if __name__ == "__main__":
    main()
