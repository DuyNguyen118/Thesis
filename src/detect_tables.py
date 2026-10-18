import argparse
import random
from pathlib import Path

import cv2
import numpy as np

from detect_layout import DPI, binarize, find_table_lines, group_positions, render_pages

INK_DELTA = 0.02   # ô được coi là có dấu nếu mật độ mực cao hơn trung vị của hàng quá mức này


def merge_close(pos, min_gap):
    """Gộp các vị trí đường quá sát nhau (đường kẻ bị tách đôi do nhiễu)."""
    out = []
    for p in sorted(pos):
        if out and p - out[-1] < min_gap:
            out[-1] = (out[-1] + p) // 2
        else:
            out.append(p)
    return out


def find_tables(bw, dpi=DPI):
    """Tách từng bảng: gộp đường ngang + dọc, mỗi khối nối liền là một bảng.

    Nét dọc của logo, gạch chân chữ... nằm ở khối riêng, nhỏ hoặc chỉ có một chiều nên bị loại.
    """
    mm = dpi / 25.4
    horiz, vert = find_table_lines(bw, dpi)
    lines = cv2.dilate(cv2.bitwise_or(horiz, vert), np.ones((5, 5), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(lines, connectivity=8)
    tables = []
    for i in range(1, n):
        x, y, w, h, _ = stats[i]
        if w < 30 * mm or h < 12 * mm:
            continue
        comp = (labels[y:y + h, x:x + w] == i).astype(np.uint8) * 255
        hz = cv2.bitwise_and(horiz[y:y + h, x:x + w], comp)
        vt = cv2.bitwise_and(vert[y:y + h, x:x + w], comp)
        rows = merge_close([p + y for p in group_positions(hz, axis=1)], int(1.5 * mm))
        cols = merge_close([p + x for p in group_positions(vt, axis=0)], int(1.5 * mm))
        if len(rows) < 3 or len(cols) < 3:
            continue
        tables.append({"x": x, "y": y, "w": w, "h": h, "rows": rows, "cols": cols})
    tables.sort(key=lambda t: t["y"])
    return tables


def cell_ink(bw, box, margin_ratio=0.15, min_margin=6):
    """Mật độ mực trong ô, đã thu nhỏ vào trong để không tính đường viền."""
    x0, y0, x1, y1 = box
    mx = max(min_margin, int((x1 - x0) * margin_ratio))
    my = max(min_margin, int((y1 - y0) * margin_ratio))
    roi = bw[y0 + my:y1 - my, x0 + mx:x1 - mx]
    if roi.size == 0:
        return 0.0
    return float((roi > 0).mean())


def score_box(t, r, c):
    rows, cols = t["rows"], t["cols"]
    return cols[c], rows[r], cols[c + 1], rows[r + 1]


def analyze_table(bw, t, n_score):
    """Bảng Likert: n_score cột cuối là cột điểm, hàng đầu là tiêu đề."""
    rows, cols = t["rows"], t["cols"]
    ncols = len(cols) - 1
    if ncols < n_score + 1 or len(rows) - 1 < 2:
        return None
    first = ncols - n_score
    results = []
    for r in range(1, len(rows) - 1):
        inks = [cell_ink(bw, score_box(t, r, c)) for c in range(first, ncols)]
        base = float(np.median(inks))
        marked = [i + 1 for i, v in enumerate(inks) if v - base > INK_DELTA]
        status = "ok" if len(marked) == 1 else ("none" if not marked else "multiple")
        results.append({"row": r, "inks": inks, "marked": marked, "status": status})
    return results


def draw_mark(img, box, style, rng):
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    cx = (x0 + x1) // 2 + rng.randint(-w // 10, w // 10)
    cy = (y0 + y1) // 2 + rng.randint(-h // 10, h // 10)
    r = int(min(w, h) * 0.28)
    t = 4
    if style == "cross":
        cv2.line(img, (cx - r, cy - r), (cx + r, cy + r), (0, 0, 0), t)
        cv2.line(img, (cx - r, cy + r), (cx + r, cy - r), (0, 0, 0), t)
    elif style == "tick":
        pts = np.array([[cx - r, cy], [cx - r // 3, cy + r], [cx + r, cy - r]], np.int32)
        cv2.polylines(img, [pts], False, (0, 0, 0), t)
    else:
        cv2.circle(img, (cx, cy), r, (0, 0, 0), t)


def annotate(img, t, results, n_score):
    vis = img.copy()
    cols = t["cols"]
    first = len(cols) - 1 - n_score
    for res in results:
        r = res["row"]
        for i in range(n_score):
            x0, y0, x1, y1 = score_box(t, r, first + i)
            if (i + 1) in res["marked"]:
                overlay = vis.copy()
                cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 0, 255), -1)
                vis = cv2.addWeighted(overlay, 0.25, vis, 0.75, 0)
                cv2.rectangle(vis, (x0 + 2, y0 + 2), (x1 - 2, y1 - 2), (0, 0, 255), 3)
        color = (0, 140, 0) if res["status"] == "ok" else (0, 0, 255)
        label = str(res["marked"][0]) if res["status"] == "ok" else res["status"]
        cv2.putText(vis, label, (t["x"] + t["w"] + 15, t["rows"][r + 1] - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3)
    return vis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("--score-cols", type=int, default=5, help="số cột điểm ở cuối bảng (mặc định 5)")
    ap.add_argument("--simulate", action="store_true", help="vẽ dấu giả để tự kiểm tra")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out = Path("output")
    out.mkdir(exist_ok=True)
    stem = Path(args.pdf).stem
    total = correct = 0

    for pno, img in enumerate(render_pages(args.pdf), 1):
        bw = binarize(img)
        tables = find_tables(bw)
        truth = {}
        if args.simulate:
            for ti, t in enumerate(tables):
                if analyze_table(bw, t, args.score_cols) is None:
                    continue
                first = len(t["cols"]) - 1 - args.score_cols
                for r in range(1, len(t["rows"]) - 1):
                    c = rng.randrange(args.score_cols)
                    style = rng.choice(["cross", "tick", "circle"])
                    draw_mark(img, score_box(t, r, first + c), style, rng)
                    truth[(ti, r)] = c + 1
            bw = binarize(img)

        print(f"Trang {pno}: {len(tables)} bảng")
        vis = img
        for ti, t in enumerate(tables):
            res = analyze_table(bw, t, args.score_cols)
            size = f"{len(t['rows']) - 1} hàng x {len(t['cols']) - 1} cột"
            if res is None:
                print(f"  Bảng {ti + 1} ({size}): không phải bảng điểm, bỏ qua")
                continue
            vis = annotate(vis, t, res, args.score_cols)
            summary = []
            for x in res:
                got = x["marked"][0] if x["status"] == "ok" else x["status"]
                if args.simulate:
                    want = truth[(ti, x["row"])]
                    total += 1
                    correct += int(x["status"] == "ok" and x["marked"][0] == want)
                    summary.append(f"{got}" if got == want else f"{got}(đúng là {want})")
                else:
                    summary.append(str(got))
            print(f"  Bảng {ti + 1} ({size}), điểm từng hàng: " + ", ".join(summary))
        cv2.imwrite(str(out / f"{stem}_p{pno}_tables.png"), vis)

    if args.simulate and total:
        print(f"\nMô phỏng: nhận đúng {correct}/{total} hàng")


if __name__ == "__main__":
    main()