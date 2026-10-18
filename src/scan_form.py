"""Quét một form (PDF/ảnh scan) và in ra dữ liệu thô nhận diện được. Chưa đối chiếu gì cả.

Với mỗi trang:
  - TEXT:      toàn bộ chữ OCR được
  - CHECKBOX:  từng ô vuông, chữ nằm bên phải ô, ô có được đánh dấu hay không
  - TABLE:     từng bảng kẻ ô, chữ ở các cột nhãn, cột nào được đánh dấu

Chạy:
    python src/scan_form.py data/samples/M02_Intern-evaluation-template.pdf
    python src/scan_form.py data/samples/M03_LABOUR-MARKET-FEEDBACK.pdf --simulate
--simulate: tự vẽ dấu giả lên ngẫu nhiên một số ô rồi kiểm tra máy nhận có đúng không.
"""
import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import pytesseract

from detect_layout import binarize, find_checkboxes, find_table_lines, render_pages
from detect_tables import draw_mark, find_tables

BOX_DELTA = 0.06     # ô vuông có dấu nếu mực trong ô cao hơn ô trống bấy nhiêu
CELL_DELTA = 0.03    # ô bảng có dấu nếu mực vùng giữa ô cao hơn ô trống cùng hàng bấy nhiêu


def remove_lines(img, bw):
    """Xóa đường kẻ bảng/gạch chân khỏi ảnh (đường kẻ làm OCR đọc chữ trong bảng rất tệ)."""
    horiz, vert = find_table_lines(bw)
    mask = cv2.dilate(cv2.bitwise_or(horiz, vert), np.ones((3, 3), np.uint8))
    clean = img.copy()
    clean[mask > 0] = (255, 255, 255)
    return clean


def rotate_img(img, deg):
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255))


def estimate_skew(bw):
    """Góc nghiêng (độ) của trang, ước lượng từ các đường thẳng dài gần nằm ngang (đường kẻ bảng, dòng chữ)."""
    h, w = bw.shape
    small = cv2.resize(bw, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    lines = cv2.HoughLinesP(small, 1, np.pi / 720, threshold=100,
                            minLineLength=int(w * 0.5 * 0.2), maxLineGap=10)
    if lines is None:
        return 0.0
    angs = []
    for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):
        a = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if abs(a) < 10:
            angs.append(a)
    return float(np.median(angs)) if len(angs) >= 3 else 0.0


def deskew(img):
    ang = estimate_skew(binarize(img))
    if abs(ang) < 0.05:
        return img, 0.0
    return rotate_img(img, ang), ang


def ocr_lang():
    """Dùng thêm tiếng Việt nếu Tesseract đã cài gói 'vie' (để đọc đúng dấu trong tên người)."""
    try:
        return "eng+vie" if "vie" in pytesseract.get_languages() else "eng"
    except Exception:
        return "eng"


def ocr(img, bw=None):
    if bw is not None:
        img = remove_lines(img, bw)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cfg = "--psm 6"
    text = pytesseract.image_to_string(gray, lang=ocr_lang(), config=cfg)
    d = pytesseract.image_to_data(gray, lang=ocr_lang(), config=cfg, output_type=pytesseract.Output.DICT)
    words = [{"text": t.strip(), "x": d["left"][i], "y": d["top"][i], "w": d["width"][i],
              "h": d["height"][i]} for i, t in enumerate(d["text"])
             if t.strip() and float(d["conf"][i]) >= 0]
    return text, words


def text_right_of(box, words):
    """Chữ cùng dòng, bên phải ô."""
    cy = box["y"] + box["h"] / 2
    sel = [w for w in words if abs(w["y"] + w["h"] / 2 - cy) < 0.8 * box["h"]
           and w["x"] >= box["x"] + box["w"] - 4]
    sel.sort(key=lambda w: w["x"])
    out, last = [], None
    for w in sel:
        if last is not None and w["x"] - last > 3 * box["h"]:
            break
        out.append(w["text"])
        last = w["x"] + w["w"]
    return " ".join(out)


def text_in(words, x0, y0, x1, y1):
    sel = [w for w in words if x0 <= w["x"] + w["w"] / 2 <= x1 and y0 <= w["y"] + w["h"] / 2 <= y1]
    sel.sort(key=lambda w: w["y"] + w["h"] / 2)
    lines, cur, cur_cy = [], [], None
    for w in sel:  # gom thành dòng theo tâm dọc, rồi sắp từng dòng từ trái sang phải
        cy = w["y"] + w["h"] / 2
        if cur and cy - cur_cy > 0.6 * w["h"]:
            lines.append(cur)
            cur = []
        cur.append(w)
        cur_cy = cy if len(cur) == 1 else cur_cy
    if cur:
        lines.append(cur)
    return " ".join(w["text"] for ln in lines for w in sorted(ln, key=lambda w: w["x"]))


def center_ink(bw, x0, y0, x1, y1, ratio=0.5):
    w, h = x1 - x0, y1 - y0
    mx, my = int(w * (1 - ratio) / 2), int(h * (1 - ratio) / 2)
    roi = bw[y0 + my:y1 - my, x0 + mx:x1 - mx]
    return float((roi > 0).mean()) if roi.size else 0.0


TM_MARGIN = 4        # viền thêm quanh ô khi cắt mẫu
TM_LOW = 0.55        # ngưỡng thấp hơn, chỉ chấp nhận nếu khung ô còn thấy rõ (ô bị nét vẽ che bớt viền)
TM_THRESHOLD = 0.75  # độ giống viền ô tối thiểu để coi là ô vuông (ô thật >= 0.85, chữ cái <= 0.61)


def ink_fill(bw, x, y, w, h):
    """Mật độ mực trong lòng ô (bỏ phần viền), để bắt cả dấu khoanh tròn/tick sát viền."""
    m = max(4, int(round(0.17 * w)))
    inner = bw[y + m: y + h - m, x + m: x + w - m]
    return float(inner.mean()) / 255.0 if inner.size else 0.0


HALO_DELTA = 0.01    # mực ở vùng quanh ô (trái/trên/dưới) cao hơn ô khác bấy nhiêu -> dấu tràn/khoanh ngoài ô


def halo_fill(bw, b):
    """Mật độ mực ở vành quanh ô (bỏ bên phải vì là chữ nhãn), để bắt dấu khoanh/tràn ra ngoài ô."""
    x, y, w, h = b["x"], b["y"], b["w"], b["h"]
    p, g = int(0.7 * w), int(0.15 * w)
    y0, x0 = max(0, y - p), max(0, x - p)
    m = np.zeros(bw.shape, np.uint8)
    m[y0:y + h + p, x0:x + w + g] = 1
    m[max(0, y - g):y + h + g, max(0, x - g):x + w + g] = 0
    return float((bw[m > 0] > 0).mean())


def _make_template(crops, w, h):
    m = TM_MARGIN
    stack = []
    for bw, x, y, cw, ch in crops:
        crop = bw[max(0, y - m):y + ch + m, max(0, x - m):x + cw + m]
        if crop.size:
            stack.append(cv2.resize(crop, (w + 2 * m, h + 2 * m)))
    tpl = np.median(np.stack(stack), axis=0).astype(np.float32)
    # chỉ so khớp phần viền ô (cộng 2px mỗi bên), bỏ qua ruột ô là nơi người ta vẽ dấu
    ring = cv2.dilate((tpl > 127).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(np.float32)
    return {"img": tpl, "mask": ring, "w": w, "h": h, "n": len(stack)}


def build_template(bws):
    """Học hình ô trống từ các ô sạch (nhận ra bằng hình dạng) của toàn bộ tài liệu.

    Một tài liệu có thể dùng nhiều kiểu ký tự ô (to/nhỏ, viền dày/mảnh) nên học một mẫu cho mỗi nhóm kích thước.
    Trả về danh sách mẫu (rỗng nếu chưa đủ ô để học).
    """
    items = []
    for bw in bws:
        for (x, y, w, h, _) in find_checkboxes(bw):
            dens = float((bw[y:y + h, x:x + w] > 0).mean())   # nét viền dày -> mật độ mực cao
            items.append((w, h, bw, x, y, dens))
    items.sort(key=lambda t: t[0])
    by_size, cur = [], []
    for it in items:   # nhóm các ô có chiều rộng chênh nhau không quá ~12%
        if cur and it[0] > cur[0][0] * 1.12:
            by_size.append(cur)
            cur = []
        cur.append(it)
    if cur:
        by_size.append(cur)
    groups = []
    for g in by_size:  # cùng kích thước nhưng viền dày/mảnh khác nhau thì tách thành mẫu riêng
        g = sorted(g, key=lambda t: t[5])
        sub = []
        for it in g:
            if sub and it[5] > sub[0][5] * 1.3:
                groups.append(sub)
                sub = []
            sub.append(it)
        if sub:
            groups.append(sub)
    tpls = []
    for g in groups:
        if len(g) < 3:
            continue
        w = int(np.median([t[0] for t in g]))
        h = int(np.median([t[1] for t in g]))
        tpls.append(_make_template([(t[2], t[3], t[4], t[0], t[1]) for t in g], w, h))
    return tpls


def sides_present(bw, x, y, w, h, need=3, frac=0.8, band=3):
    """Khung ô còn nguyên: ít nhất `need` trong 4 cạnh có mực chạy gần hết chiều dài (chữ cái không có)."""
    H, W = bw.shape
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    if x1 - x0 < w * 0.8 or y1 - y0 < h * 0.8:
        return False
    def side(sl, axis):
        return float((sl.max(axis=axis) > 0).mean()) >= frac
    top = bw[y0:y0 + band * 2, x0:x1]       # dải ngang sát cạnh trên
    bot = bw[y1 - band * 2:y1, x0:x1]
    left = bw[y0:y1, x0:x0 + band * 2]
    right = bw[y0:y1, x1 - band * 2:x1]
    return sum([side(top, 0), side(bot, 0), side(left, 1), side(right, 1)]) >= need


def _ring_ink(bw, x, y, w, h, gap=3, thick=5):
    """Mật độ mực ở vành ngay bên ngoài khung ô."""
    H, W = bw.shape
    x0, y0, x1, y1 = max(0, x - gap - thick), max(0, y - gap - thick), min(W, x + w + gap + thick), min(H, y + h + gap + thick)
    big = bw[y0:y1, x0:x1].copy()
    ix0, iy0 = max(0, x - gap) - x0, max(0, y - gap) - y0
    big[iy0:iy0 + h + 2 * gap, ix0:ix0 + w + 2 * gap] = 0
    return float((big > 0).mean())


def frame_ok(bw, x, y, w, h, need_sides=3, need_corners=3):
    """Giống khung ô thật: cạnh thẳng, góc vuông có mực, quanh ô trống (loại chữ D, O, cụm chữ...)."""
    if not sides_present(bw, x, y, w, h, need=need_sides, frac=0.9):
        return False
    H, W = bw.shape
    pts = [(x, y), (x + w - 4, y), (x, y + h - 4), (x + w - 4, y + h - 4)]
    ok = sum(float((bw[max(0, py):py + 4, max(0, px):px + 4] > 0).mean()) > 0.3 for px, py in pts)
    return ok >= need_corners and _ring_ink(bw, x, y, w, h) <= 0.03


def find_boxes_by_lines(bw, w, h):
    """Tìm khung ô bằng 4 cạnh thẳng: bắt được ô mà nét vẽ (khoanh tròn...) che mất một phần viền."""
    hz = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (int(0.7 * w), 1)))
    vt = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(0.7 * h))))
    m = cv2.dilate(cv2.bitwise_or(hz, vt), np.ones((3, 3), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    out = []
    for i in range(1, n):
        x, y, cw, ch, _ = stats[i]
        x, y, cw, ch = x + 1, y + 1, cw - 2, ch - 2   # bỏ phần nở của dilate
        if not (abs(cw - w) <= 0.08 * w and abs(ch - h) <= 0.08 * h):
            continue
        if not frame_ok(bw, x, y, cw, ch, need_sides=4, need_corners=4):
            continue
        out.append((x, y, cw, ch))
    return out


def find_boxes_by_template(bw, tpls):
    """Tìm ô vuông bằng khớp mẫu: vẫn nhận ra ô khi có nét tick/gạch chéo đè lên viền."""
    cand = []
    bwf = bw.astype(np.float32)
    shape_boxes = find_checkboxes(bw)
    for ti, tpl in enumerate(tpls):
        res = 1.0 - cv2.matchTemplate(bwf, tpl["img"], cv2.TM_SQDIFF_NORMED, mask=tpl["mask"])
        ys, xs = np.where(res >= TM_LOW)
        for px, py in zip(xs, ys):
            cand.append((float(res[py, px]), int(px) + TM_MARGIN + tpl["w"] / 2,
                         int(py) + TM_MARGIN + tpl["h"] / 2, ti))
    cand.sort(key=lambda c: -c[0])
    kept = []
    for score, cx, cy, ti in cand:  # giữ điểm khớp tốt nhất trong mỗi vùng (non-maximum suppression)
        t = tpls[ti]
        if score < TM_THRESHOLD and not frame_ok(
                bw, int(round(cx - t["w"] / 2)), int(round(cy - t["h"] / 2)), t["w"], t["h"]):
            continue
        d = 0.7 * tpls[ti]["w"]
        if all((cx - kx) ** 2 + (cy - ky) ** 2 >= min(d, 0.7 * tpls[kt]["w"]) ** 2
               for _, kx, ky, kt in kept):
            kept.append((score, cx, cy, ti))
    boxes = []
    for score, cx, cy, ti in kept:
        t = tpls[ti]
        x, y = int(round(cx - t["w"] / 2)), int(round(cy - t["h"] / 2))
        boxes.append({"x": x, "y": y, "w": t["w"], "h": t["h"], "tpl": ti,
                      "fill": ink_fill(bw, x, y, t["w"], t["h"]), "match": round(score, 2)})
    # ô mà khớp mẫu bỏ sót vẫn được giữ lại: ô sạch nhận ra bằng hình dạng, ô bị nét vẽ che viền nhận ra bằng 4 cạnh
    extra = [(int(x), int(y), int(w), int(h)) for (x, y, w, h, _) in shape_boxes]
    for t in (tpls if len(boxes) >= 2 else []):   # trang gần như không có ô thì không tìm thêm
        extra += [(int(a), int(b_), int(c), int(d)) for a, b_, c, d in find_boxes_by_lines(bw, t["w"], t["h"])]
    for (x, y, w, h) in extra:
        if not any(abs(b["x"] - x) < 0.5 * w and abs(b["y"] - y) < 0.5 * h for b in boxes):
            boxes.append({"x": x, "y": y, "w": w, "h": h, "tpl": -1,
                          "fill": ink_fill(bw, x, y, w, h), "match": 0.0})
    boxes.sort(key=lambda b: (b["y"], b["x"]))
    return boxes


def scan_boxes(bw, words, tpls):
    if tpls:
        boxes = find_boxes_by_template(bw, tpls)
    else:
        boxes = [{"x": x, "y": y, "w": w, "h": h, "fill": f, "tpl": 0}
                 for (x, y, w, h, f) in find_checkboxes(bw)]
    if not boxes:
        return boxes, 0.0
    for b in boxes:
        b["halo"] = halo_fill(bw, b)
    allf = float(np.median([b["fill"] for b in boxes]))
    allh = float(np.median([b["halo"] for b in boxes]))
    for b in boxes:   # mức "ô trống" tính riêng cho từng kiểu ô (ô nhỏ/mảnh có mực khác ô to)
        grp = [o for o in boxes if o["tpl"] == b["tpl"]]
        base = float(np.median([o["fill"] for o in grp])) if len(grp) >= 3 else allf
        hbase = float(np.median([o["halo"] for o in grp])) if len(grp) >= 3 else allh
        b["marked"] = b["fill"] > base + BOX_DELTA or b["halo"] > hbase + HALO_DELTA
        b["text"] = text_right_of(b, words)
    return boxes, allf


def scan_tables(bw, words, n_score):
    out = []
    for ti, t in enumerate(find_tables(bw)):
        rows, cols = t["rows"], t["cols"]
        ncol = len(cols) - 1
        if ncol <= n_score or len(rows) < 3:
            continue
        first = ncol - n_score
        header = [text_in(words, cols[c], rows[0], cols[c + 1], rows[1]) for c in range(first, ncol)]
        data = []
        for r in range(1, len(rows) - 1):
            inks = [center_ink(bw, cols[c], rows[r], cols[c + 1], rows[r + 1])
                    for c in range(first, ncol)]
            base = float(np.median(inks))
            marked = [i for i, v in enumerate(inks) if v - base > CELL_DELTA]
            data.append({"row": r,
                         "label": text_in(words, cols[0], rows[r], cols[first], rows[r + 1]),
                         "inks": [round(v, 3) for v in inks], "marked_cols": marked,
                         "cells": [(cols[c], rows[r], cols[c + 1], rows[r + 1])
                                   for c in range(first, ncol)]})
        out.append({"index": ti + 1, "header": header, "rows": data, "bbox": (t["x"], t["y"], t["w"], t["h"])})
    return out


def learn_template(images):
    return build_template([binarize(im) for im in images])


def scan_page(img, n_score, tpl=None):
    bw = binarize(img)
    text, words = ocr(img, bw)
    boxes, base = scan_boxes(bw, words, tpl)
    tables = scan_tables(bw, words, n_score)
    return {"text": text, "words": words, "boxes": boxes, "box_base": base, "tables": tables}


def annotate(img, page):
    vis = img.copy()
    for b in page["boxes"]:
        col = (0, 0, 255) if b["marked"] else (0, 160, 0)
        if b["marked"]:
            ov = vis.copy()
            cv2.rectangle(ov, (b["x"] - 4, b["y"] - 4), (b["x"] + b["w"] + 4, b["y"] + b["h"] + 4), col, -1)
            vis = cv2.addWeighted(ov, 0.3, vis, 0.7, 0)
        cv2.rectangle(vis, (b["x"] - 4, b["y"] - 4), (b["x"] + b["w"] + 4, b["y"] + b["h"] + 4), col, 3)
    for t in page["tables"]:
        for r in t["rows"]:
            for i in r["marked_cols"]:
                x0, y0, x1, y1 = r["cells"][i]
                ov = vis.copy()
                cv2.rectangle(ov, (x0, y0), (x1, y1), (0, 0, 255), -1)
                vis = cv2.addWeighted(ov, 0.25, vis, 0.75, 0)
    return vis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("--score-cols", type=int, default=5)
    ap.add_argument("--simulate", action="store_true")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--overshoot", type=float, default=0.0, help="xác suất dấu tràn ra ngoài ô (mô phỏng)")
    ap.add_argument("--mixed", action="store_true", help="mô phỏng tài liệu dùng lẫn ô nhỏ viền mảnh (như bản Word thật)")
    ap.add_argument("--rotate", type=float, default=0.0, help="xoay trang vài độ (mô phỏng scan lệch)")
    args = ap.parse_args()

    out = Path("output")
    out.mkdir(exist_ok=True)
    stem = Path(args.pdf).stem
    images = list(render_pages(args.pdf))
    rng = random.Random(args.seed)
    truth = []  # (trang, loại, ...) những ô đã vẽ dấu giả

    if args.simulate and args.mixed:
        mrng = random.Random(args.seed + 100)
        for img in images:
            for (x, y, w, h, _) in find_checkboxes(binarize(img)):
                r = mrng.random()
                if r < 0.5:
                    cv2.rectangle(img, (x - 3, y - 3), (x + w + 3, y + h + 3), (255, 255, 255), -1)
                    n = int(w * 0.65) if r < 0.25 else w   # ô nhỏ, hoặc cùng cỡ nhưng viền mảnh
                    x0, y0 = x + (w - n) // 2, y + (h - n) // 2
                    cv2.rectangle(img, (x0, y0), (x0 + n, y0 + n), (0, 0, 0), 2 if r < 0.25 else 1)

    if args.simulate:
        tpl0 = learn_template(images)
        for pno, img in enumerate(images, 1):
            page = scan_page(img, args.score_cols, tpl0)
            for bi, b in enumerate(page["boxes"]):
                if rng.random() < 0.3:
                    k = 1.7 if rng.random() < args.overshoot else 0.5   # 1.7 -> nét vẽ tràn ra ngoài ô
                    cx, cy = b["x"] + b["w"] / 2, b["y"] + b["h"] / 2
                    box = (int(cx - k * b["w"]), int(cy - k * b["h"]), int(cx + k * b["w"]), int(cy + k * b["h"]))
                    draw_mark(img, box, rng.choice(["cross", "tick", "circle"]), rng)
                    truth.append((pno, "box", (b["x"], b["y"])))
            for t in page["tables"]:
                for r in t["rows"]:
                    c = rng.randrange(len(r["cells"]))
                    draw_mark(img, r["cells"][c], rng.choice(["cross", "tick", "circle"]), rng)
                    truth.append((pno, "cell", (t["index"], r["row"], c)))

    if args.simulate and args.rotate:
        images = [rotate_img(im, args.rotate) for im in images]

    skews = []
    fixed = []
    for im in images:  # tự xoay thẳng lại từng trang trước khi quét
        im2, ang = deskew(im)
        fixed.append(im2)
        skews.append(ang)
    images = fixed
    if any(skews):
        print("Góc nghiêng đã chỉnh (độ):", [round(a, 2) for a in skews])

    results = []
    tpl = learn_template(images)  # học hình ô trống từ chính tài liệu đang quét
    if tpl:
        print("Mẫu ô vuông học được:", ", ".join(f"{t['w']}x{t['h']} px ({t['n']} ô)" for t in tpl))
    for pno, img in enumerate(images, 1):
        page = scan_page(img, args.score_cols, tpl)
        results.append(page)
        print(f"\n=================== TRANG {pno} ===================")
        print("[TEXT]")
        print(page["text"].strip() or "(không có chữ)")
        print(f"\n[CHECKBOX] {len(page['boxes'])} ô, {sum(b['marked'] for b in page['boxes'])} ô có dấu")
        for b in page["boxes"]:
            print(f"  {'[X]' if b['marked'] else '[ ]'} (x={b['x']}, y={b['y']}) {b['text']}")
        for t in page["tables"]:
            print(f"\n[TABLE {t['index']}] cột điểm: {t['header']}")
            for r in t["rows"]:
                chosen = ", ".join(t["header"][i] or str(i + 1) for i in r["marked_cols"]) or "(không có)"
                print(f"  hàng {r['row']}: {r['label'][:60]!r} -> đánh dấu: {chosen}")
        cv2.imwrite(str(out / f"{stem}_p{pno}_scan.png"), annotate(img, page))

    if args.simulate:
        ok = bad = 0
        for pno, kind, key in truth:
            page = results[pno - 1]
            if kind == "box":
                hit = [b for b in page["boxes"]
                       if abs(b["x"] - key[0]) < 0.5 * b["w"] and abs(b["y"] - key[1]) < 0.5 * b["h"]]
                good = bool(hit) and hit[0]["marked"]
            else:
                ti, r, c = key
                t = next((t for t in page["tables"] if t["index"] == ti), None)
                row = next((x for x in t["rows"] if x["row"] == r), None) if t else None
                good = bool(row) and row["marked_cols"] == [c]
            ok += good
            bad += not good
            if not good:
                print(f"  SAI: trang {pno} {kind} {key}")
        # ô không vẽ dấu mà máy lại báo có dấu (báo nhầm)
        drawn = [(p, k) for p, kind, k in truth if kind == "box"]
        fps = [(pno, b) for pno, page in enumerate(results, 1) for b in page["boxes"]
               if b["marked"] and not any(p == pno and abs(b["x"] - k[0]) < 0.5 * b["w"]
                                          and abs(b["y"] - k[1]) < 0.5 * b["h"] for p, k in drawn)]
        for pno, b in fps:
            print(f"  BÁO NHẦM: trang {pno} ô (x={b['x']}, y={b['y']}) fill={b['fill']:.2f} halo={b['halo']:.3f} {b['text'][:30]!r}")
        false_pos = len(fps)
        print(f"\nMô phỏng: nhận đúng {ok}/{ok + bad} dấu đã vẽ, báo nhầm {false_pos} ô trống")

    with open(out / f"{stem}_scan.json", "w", encoding="utf-8") as f:
        json.dump([{"text": p["text"], "boxes": p["boxes"],
                    "tables": [{**t, "rows": [{k: v for k, v in r.items() if k != "cells"}
                                              for r in t["rows"]]} for t in p["tables"]]}
                   for p in results], f, ensure_ascii=False, indent=2, default=str)


if __name__ == "__main__":
    main()