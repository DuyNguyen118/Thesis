"""Chấm kết quả quét so với đáp án: output/<stem>_scan.json vs data/truth/<stem>.json.

Ô checkbox ghép theo vị trí (cùng trang, góc trên-trái lệch <= BOX_TOL * cao ô, lấy cặp gần nhất).
Bảng ghép theo bbox (IoU >= 0.5), hàng ghép theo số thứ tự hàng.

In ra:
  - ô: tìm được / sót / thừa
  - dấu ở ô: đúng (tp) / sót (fn: ô có dấu mà máy báo trống hoặc không tìm ra ô) / báo nhầm (fp)
  - hàng bảng: số hàng đánh dấu đúng cột / tổng
  - nhãn: độ giống trung bình (RapidFuzz) giữa nhãn OCR và nhãn đáp án

Chạy:
    python src/evaluate.py                                # mọi đáp án trong data/truth với output/
    python src/evaluate.py output/Data1.1_scan.json       # một file
"""
import argparse
import json
import sys
from pathlib import Path

from rapidfuzz import fuzz

BOX_TOL = 1.2        # ô máy tìm và ô đáp án lệch nhau quá 1.2 * cao ô thì coi là 2 ô khác nhau
LABEL_BAD = 80       # nhãn giống đáp án dưới mức này (0-100) bị liệt kê


def _iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    return inter / float(aw * ah + bw * bh - inter) if inter else 0.0


def _match_boxes(pred, truth):
    """Ghép tham lam theo khoảng cách tăng dần. Trả về [(i_pred, i_truth)]."""
    pairs = []
    for i, p in enumerate(pred):
        for j, t in enumerate(truth):
            d = ((p["x"] - t["x"]) ** 2 + (p["y"] - t["y"]) ** 2) ** 0.5
            if d <= BOX_TOL * t["h"]:
                pairs.append((d, i, j))
    pairs.sort()
    used_p, used_t, out = set(), set(), []
    for _, i, j in pairs:
        if i not in used_p and j not in used_t:
            used_p.add(i)
            used_t.add(j)
            out.append((i, j))
    return out


def evaluate(scan, truth):
    """So một tài liệu. `scan` là nội dung *_scan.json, `truth` là nội dung data/truth/*.json."""
    r = {"boxes": 0, "found": 0, "missed": 0, "extra": 0, "tp": 0, "fn": 0, "fp": 0,
         "rows": 0, "rows_ok": 0, "tables_missed": 0, "label_scores": [], "errors": []}
    tpages = {p["page"]: p for p in truth["pages"]}
    if len(scan) != len(tpages):
        r["errors"].append(f"số trang khác: máy {len(scan)}, đáp án {len(tpages)}")
    for pno in sorted(set(range(1, len(scan) + 1)) | set(tpages)):
        pp = scan[pno - 1] if pno <= len(scan) else {"boxes": [], "tables": []}
        tp = tpages.get(pno, {"boxes": [], "tables": []})

        pairs = _match_boxes(pp["boxes"], tp["boxes"])
        mp = {i for i, _ in pairs}
        mt = {j for _, j in pairs}
        r["boxes"] += len(tp["boxes"])
        r["found"] += len(pairs)
        for i, j in pairs:
            pb, tb = pp["boxes"][i], tp["boxes"][j]
            if tb["marked"] and pb["marked"]:
                r["tp"] += 1
            elif tb["marked"]:
                r["fn"] += 1
                r["errors"].append(f"trang {pno} SÓT DẤU ô (x={tb['x']}, y={tb['y']}) {tb['label'][:40]!r}")
            elif pb["marked"]:
                r["fp"] += 1
                r["errors"].append(f"trang {pno} BÁO NHẦM ô (x={tb['x']}, y={tb['y']}) {tb['label'][:40]!r}")
            if tb.get("label"):
                s = fuzz.ratio(pb.get("text", ""), tb["label"])
                r["label_scores"].append(s)
                if s < LABEL_BAD:
                    r["errors"].append(f"trang {pno} NHÃN {s:.0f}: {pb.get('text', '')[:40]!r} != {tb['label'][:40]!r}")
        for j, tb in enumerate(tp["boxes"]):
            if j not in mt:
                r["missed"] += 1
                r["fn"] += tb["marked"]
                r["errors"].append(f"trang {pno} SÓT Ô{' CÓ DẤU' if tb['marked'] else ''} "
                                   f"(x={tb['x']}, y={tb['y']}) {tb['label'][:40]!r}")
        for i, pb in enumerate(pp["boxes"]):
            if i not in mp:
                r["extra"] += 1
                r["fp"] += bool(pb["marked"])
                r["errors"].append(f"trang {pno} Ô THỪA{' CÓ DẤU' if pb['marked'] else ''} "
                                   f"(x={pb['x']}, y={pb['y']}) {pb.get('text', '')[:40]!r}")

        for tt in tp["tables"]:
            r["rows"] += len(tt["rows"])
            cand = [(_iou([int(v) for v in t["bbox"]], tt["bbox"]), t) for t in pp["tables"]]
            iou, pt = max(cand, key=lambda c: c[0], default=(0.0, None))
            if pt is None or iou < 0.5:
                r["tables_missed"] += 1
                r["errors"].append(f"trang {pno} SÓT BẢNG {tt['index']} ({len(tt['rows'])} hàng)")
                continue
            prow = {x["row"]: x for x in pt["rows"]}
            for tr in tt["rows"]:
                got = prow.get(tr["row"], {}).get("marked_cols")
                if got == tr["marked_cols"]:
                    r["rows_ok"] += 1
                else:
                    r["errors"].append(f"trang {pno} BẢNG {tt['index']} hàng {tr['row']}: "
                                       f"máy {got}, đúng {tr['marked_cols']} {tr['label'][:30]!r}")
    sc = r.pop("label_scores")
    r["label"] = round(sum(sc) / len(sc), 1) if sc else None
    return r


def summary(name, r):
    lab = f", nhãn {r['label']}" if r["label"] is not None else ""
    return (f"{name}: ô {r['found']}/{r['boxes']} (sót {r['missed']}, thừa {r['extra']}) | "
            f"dấu đúng {r['tp']}, sót {r['fn']}, báo nhầm {r['fp']} | "
            f"hàng bảng {r['rows_ok']}/{r['rows']}{lab}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scans", nargs="*", help="output/<stem>_scan.json (mặc định: mọi đáp án trong --truth-dir)")
    ap.add_argument("--truth-dir", default="data/truth")
    ap.add_argument("--quiet", action="store_true", help="chỉ in dòng tổng kết")
    args = ap.parse_args()

    tdir = Path(args.truth_dir)
    scans = [Path(s) for s in args.scans] or [Path("output") / f"{t.stem}_scan.json" for t in sorted(tdir.glob("*.json"))]
    bad = False
    for s in scans:
        stem = s.name.removesuffix("_scan.json")
        tpath = tdir / f"{stem}.json"
        if not tpath.exists() or not s.exists():
            print(f"{stem}: thiếu {'đáp án ' + str(tpath) if not tpath.exists() else 'kết quả ' + str(s)}")
            bad = True
            continue
        with open(s, encoding="utf-8") as f:
            scan = json.load(f)
        with open(tpath, encoding="utf-8") as f:
            truth = json.load(f)
        r = evaluate(scan, truth)
        print(summary(stem, r) + ("" if truth.get("verified") else "  [CẢNH BÁO: đáp án chưa verified]"))
        if not args.quiet:
            for e in r["errors"]:
                print("   ", e)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
