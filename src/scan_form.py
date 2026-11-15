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
from common import TRUTH

from detect_layout import binarize, find_checkboxes, find_table_lines, render_pages
from detect_tables import draw_mark, find_tables

BOX_DELTA = 0.06     # ô vuông có dấu nếu mực trong ô cao hơn ô trống bấy nhiêu
CELL_DELTA = 0.03    # ô bảng có dấu nếu mực vùng giữa ô cao hơn ô trống cùng hàng bấy nhiêu
CELL_FAINT = 0.012   # không ô nào qua CELL_DELTA: lấy ô đậm nhất nếu cao hơn trung vị bấy nhiêu (tick mảnh, Data9.1 +0.014)
CELL_LEAD = 0.5      # ... và vượt ô đậm thứ hai >= 50% khoảng cách tới trung vị (bóng mờ ở ô bên cạnh, Data6.1 +0.011)
CELL_CLEAN = 0.002   # hàng sạch (PDF điện tử, ô khác ~0 mực) thì chỉ cần cao hơn CELL_CLEAN_MIN (Data2.2 +0.008)
CELL_CLEAN_MIN = 0.005


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
    if abs(ang) < 0.45:
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
    right = box["x"] + box["w"]
    sel = [w for w in words if abs(w["y"] + w["h"] / 2 - cy) < 0.8 * box["h"]
           and w["x"] + w["w"] / 2 > right + 0.5 * box["w"] and w["x"] + w["w"] > right]
    sel.sort(key=lambda w: w["x"])
    out, last = [], None
    for w in sel:
        if last is not None and w["x"] - last > 3 * box["h"]:
            break
        txt = w["text"]
        if w["x"] < right - 4:
            # OCR gộp ô + dấu tick vào từ đầu nhãn ("WVery", "WAll"): bỏ các ký tự đè lên ô, nhưng chỉ cắt
            # tới chữ in hoa gần nhất (nhãn luôn bắt đầu bằng chữ hoa) để không cắt nhầm "Very" -> "ery"
            n = max(1, int(round(len(txt) * (right - w["x"]) / w["w"])))
            ups = [i for i in range(min(n, len(txt) - 1) + 1) if txt[i].isupper()]
            if ups:
                txt = txt[ups[-1]:]
        if txt:
            out.append(txt)
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
TM_FRAME = 0.60      # khớp dưới mức này (ô in mảnh/nhạt hơn mẫu, Data9.1 trang 7) phải còn >= 3 cạnh thẳng mới là ô
TM_THRESHOLD = 0.75  # độ giống viền ô tối thiểu để coi là ô vuông (ô thật >= 0.85, chữ cái <= 0.61)
IN_WORD_SHIFT = 0.6  # ô lệch vào trong từ OCR >= 0.6 * chiều rộng ô mới bị coi là chữ cái (cũ: 0.3 -> sót ô có tick)
NEIGHBOR_GAP = 3.0   # 2 ô cùng dòng cách nhau < 3 * rộng ô -> ô bên phải là chữ cái đầu nhãn (ô bảng cách ~270px)
WORD_MAX_H = 1.5    # "từ" OCR cao hơn 1.5 * trung vị chiều cao từ là cụm dấu vẽ tay (66-82px vs chữ ~40px)


def ink_fill(bw, x, y, w, h):
    """Mật độ mực trong lòng ô (bỏ phần viền), để bắt cả dấu khoanh tròn/tick sát viền."""
    m = max(5, int(round(0.24 * min(w, h))))
    inner = bw[y + m: y + h - m, x + m: x + w - m]
    return float(inner.mean()) / 255.0 if inner.size else 0.0


HALO_DELTA = 0.01    # mực ở vùng quanh ô (trái/trên/dưới) cao hơn ô khác bấy nhiêu -> dấu tràn/khoanh ngoài ô
HALO_ONLY = 0.06     # vành quanh ô nhiều mực dù ruột trống -> khoanh tròn to bao ngoài ô (Data1.1: ô trống halo <= 0.005)


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


def build_template(bws, words_list=None):
    """Học hình ô trống từ các ô sạch (nhận ra bằng hình dạng) của toàn bộ tài liệu.

    Một tài liệu có thể dùng nhiều kiểu ký tự ô (to/nhỏ, viền dày/mảnh) nên học một mẫu cho mỗi nhóm kích thước.
    Trả về danh sách mẫu (rỗng nếu chưa đủ ô để học).
    """
    items = []
    for i, bw in enumerate(bws):
        w_list = words_list[i] if words_list and i < len(words_list) else None
        for (x, y, w, h, _) in find_checkboxes(bw, words=w_list):
            dens = float((bw[y:y + h, x:x + w] > 0).mean())   # nét viền dày -> mật độ mực cao
            items.append((w, h, bw, x, y, dens))
    if not items:
        return []
    items.sort(key=lambda t: t[0])
    by_size, cur = [], []
    for it in items:   # nhóm các ô có chiều rộng chênh nhau không quá ~15%
        if cur and it[0] > cur[0][0] * 1.15:
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
    if not sides_present(bw, x, y, w, h, need=need_sides, frac=0.8):
        return False
    H, W = bw.shape
    pts = [(x, y), (x + w - 4, y), (x, y + h - 4), (x + w - 4, y + h - 4)]
    ok = sum(float((bw[max(0, py):py + 4, max(0, px):px + 4] > 0).mean()) > 0.3 for px, py in pts)
    return ok >= need_corners and _ring_ink(bw, x, y, w, h) <= 0.08


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
        if not (abs(cw - w) <= 0.12 * w and abs(ch - h) <= 0.12 * h):
            continue
        if not frame_ok(bw, x, y, cw, ch, need_sides=3, need_corners=3):
            continue
        out.append((x, y, cw, ch))
    return out


SOFT_SHARPNESS = 200  # phương sai Laplacian dưới mức này là trang mờ (Data9.1 trang 7: 122; trang rõ 280-1300)
SOFT_INK = 40         # mực in có độ xám trung vị trên mức này là trang nhạt (Data9.1 trang 4: 98; trang đậm 7-35)


def page_is_soft(img, bw):
    """Trang scan mờ hoặc mực nhạt: ô in mảnh chỉ khớp mẫu 0.55-0.60, được dùng ngưỡng TM_LOW.

    Trang rõ giữ ngưỡng TM_FRAME vì ở đó điểm khớp 0.55-0.60 hầu hết là chữ cái đầu tiêu đề (Data1.1, Data3.1).
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ink = gray[bw > 0]
    return (cv2.Laplacian(gray, cv2.CV_64F).var() < SOFT_SHARPNESS
            or (ink.size > 0 and float(np.median(ink)) > SOFT_INK))


def find_boxes_by_template(bw, tpls, words=None, low=TM_FRAME):
    """Tìm ô vuông bằng khớp mẫu: vẫn nhận ra ô khi có nét tick/gạch chéo đè lên viền.

    `low` là điểm khớp tối thiểu: TM_FRAME cho trang rõ, TM_LOW cho trang mờ (page_is_soft)."""
    cand = []
    H, W = bw.shape
    bwf = bw.astype(np.float32)
    res_maps = []
    for ti, tpl in enumerate(tpls):
        res = 1.0 - cv2.matchTemplate(bwf, tpl["img"], cv2.TM_SQDIFF_NORMED, mask=tpl["mask"])
        res = np.nan_to_num(res, nan=0.0)
        res_maps.append(res)
        ys, xs = np.where(res >= low)
        for px, py in zip(xs, ys):
            cand.append((float(res[py, px]), int(px) + TM_MARGIN + tpl["w"] / 2,
                         int(py) + TM_MARGIN + tpl["h"] / 2, ti))
    cand.sort(key=lambda c: -c[0])
    kept = []
    for score, cx, cy, ti in cand:  # giữ điểm khớp tốt nhất trong mỗi vùng (non-maximum suppression)
        t = tpls[ti]
        bx = int(round(cx - t["w"] / 2))
        by = int(round(cy - t["h"] / 2))
        if bx < 0.06 * W or bx > 0.88 * W or by < 0.04 * H or by > 0.93 * H:
            continue
        # Bỏ qua nếu là chữ cái nằm sâu bên trong một từ OCR (loại bỏ chữ cái in hoa trong tiêu đề/đoạn văn).
        # Ô có dấu tick tràn thường bị OCR gộp vào từ đầu nhãn ("WVery", "cf") và lệch vào trong từ ~7px,
        # nên chỉ loại khi mép trái ô nằm sau đầu từ >= IN_WORD_SHIFT * chiều rộng ô (chữ cái thật lệch >= 1 ký tự).
        if words and _inside_word(words, cx, cy, t["w"]):
            continue
        d = 0.7 * t["w"]
        if all((cx - kx) ** 2 + (cy - ky) ** 2 >= min(d, 0.7 * tpls[kt]["w"]) ** 2
               for _, kx, ky, kt in kept):
            if score < TM_FRAME and (not sides_present(bw, bx, by, t["w"], t["h"], need=3, frac=0.7)
                                     or (words and _starts_word(words, bx, by, t["w"], t["h"]))):
                continue
            kept.append((score, cx, cy, ti))

    # Lọc tính thẳng hàng của cột ô checkbox: loại bỏ các điểm nhiễu ngẫu nhiên
    final_kept = []
    for score, cx, cy, ti in kept:
        bx = int(round(cx - tpls[ti]["w"] / 2))
        aligned = sum(abs(int(round(other_cx - tpls[kt]["w"] / 2)) - bx) <= 25 for _, other_cx, _, kt in kept)
        if aligned >= 2 or score >= 0.80:
            final_kept.append((score, cx, cy, ti))

    boxes = []
    for score, cx, cy, ti in final_kept:
        t = tpls[ti]
        x, y = int(round(cx - t["w"] / 2)), int(round(cy - t["h"] / 2))
        boxes.append({"x": x, "y": y, "w": t["w"], "h": t["h"], "tpl": ti,
                      "fill": ink_fill(bw, x, y, t["w"], t["h"]), "match": round(score, 2)})
    # chữ cái đầu nhãn ("G" của "Generally") ngay sau ô nhỏ cũng giống ô -> cùng dòng, sát nhau thì giữ ô bên trái
    boxes = [b for b in boxes if not any(
        o is not b and abs(o["y"] - b["y"]) < 0.5 * b["h"] and 0 < b["x"] - o["x"] < NEIGHBOR_GAP * max(o["w"], b["w"])
        for o in boxes)]
    boxes += _rescue_in_columns(bw, tpls, res_maps, boxes, words)
    for b in boxes:   # số ô cùng cột (kể cả chính nó): ô lạc ngoài cột không được dùng luật halo-only
        b["column"] = sum(abs(o["x"] - b["x"]) <= 25 for o in boxes)

    boxes.sort(key=lambda b: (b["y"], b["x"]))
    return boxes


def _starts_word(words, x, y, w, h):
    """Một từ OCR dài bắt đầu ngay trong ô -> ô là chữ cái đầu của từ đó ("I" của "INITIATIVE", Data1.1).

    Chỉ dùng cho điểm khớp yếu: ô thật có dấu tick hay bị OCR gộp vào từ đầu nhãn ("WVery") nhưng khớp >= TM_FRAME.
    """
    cy = y + h / 2
    return any(len(mw["text"].strip()) >= 3 and x - 3 <= mw["x"] <= x + 0.5 * w and mw["x"] + mw["w"] > x + 1.5 * w
               and mw["y"] <= cy <= mw["y"] + mw["h"] for mw in words)


def _inside_word(words, cx, cy, bw_):
    """Tâm ô (cx, cy) nằm sâu trong một từ OCR dài -> là chữ cái, không phải ô.

    Bỏ qua "từ" do chính dấu vẽ tay tạo ra: OCR hay đọc ô + dấu tràn thành 'Wal', 'xi', '©Has'... cao gấp
    rưỡi dòng chữ bình thường (> WORD_MAX_H * trung vị chiều cao từ trên trang) hoặc kết thúc ngay sau mép phải ô.
    """
    max_h = WORD_MAX_H * float(np.median([w["h"] for w in words]))
    for mw in words:
        if mw["h"] > max_h or mw["x"] + mw["w"] < cx + bw_:
            continue
        if len(mw.get("text", "").strip()) >= 2 and mw["w"] > bw_ * 1.3:
            if (mw["x"] + bw_ * (0.5 + IN_WORD_SHIFT) <= cx <= mw["x"] + mw["w"] - 3
                    and mw["y"] - 3 <= cy <= mw["y"] + mw["h"] + 3):
                return True
    return False


TM_RESCUE = 0.40     # ngưỡng khớp thấp cho ô nằm đúng cột ô đã biết nhưng bị dấu to/tràn đè viền (đo được 0.44-0.65)
RESCUE_FILL = 0.12   # ô cứu thêm phải có dấu rõ (ruột nhiều mực hoặc halo cao) -> ô trống thật luôn đã qua ngưỡng 0.60


def _rescue_in_columns(bw, tpls, res_maps, boxes, words):
    """Cứu ô bị dấu to che viền: chỉ dò dọc theo các cột đã có >= 3 ô, chấp nhận điểm khớp >= TM_RESCUE
    nếu ô có mực (đã đánh dấu), có chữ nhãn bên phải và không nằm trong từ OCR."""
    if len(boxes) < 3:
        return []
    H, W = bw.shape
    cols, cur = [], []
    for b in sorted(boxes, key=lambda b: b["x"]):   # gom cột theo x (cho phép lệch 25px do nghiêng còn sót)
        if cur and b["x"] - cur[-1]["x"] > 25:
            cols.append(cur)
            cur = []
        cur.append(b)
    cols.append(cur)
    base = float(np.median([b["fill"] for b in boxes]))
    found = []
    for col in cols:
        if len(col) < 3:   # cột 2 ô dễ là chữ cái đầu tiêu đề + 1 ô (Data1.1 "INITIATIVE")
            continue
        for ti in range(len(tpls)):
            # thử mọi mẫu: một cột có thể trộn ô to/nhỏ (--mixed)
            t, res = tpls[ti], res_maps[ti]
            x0 = max(0, min(b["x"] for b in col) - 8 - TM_MARGIN)
            x1 = min(res.shape[1], max(b["x"] for b in col) + 9 - TM_MARGIN)
            if x1 <= x0:
                continue
            strip = res[:, x0:x1]
            best_x = strip.argmax(axis=1)
            best = strip.max(axis=1)
            for py in np.argsort(-best):
                if best[py] < TM_RESCUE:
                    break
                x, y = int(x0 + best_x[py] + TM_MARGIN), int(py + TM_MARGIN)
                if y < 0.04 * H or y > 0.93 * H:
                    continue
                if any(abs(o["y"] - y) < 0.8 * o["h"] and abs(o["x"] - x) < o["w"] for o in boxes + found):
                    continue
                cx, cy = x + t["w"] / 2, y + t["h"] / 2
                if words and _inside_word(words, cx, cy, t["w"]):
                    continue
                fill = ink_fill(bw, x, y, t["w"], t["h"])
                b = {"x": x, "y": y, "w": t["w"], "h": t["h"], "tpl": ti, "fill": fill,
                     "match": round(float(best[py]), 2), "rescued": True}
                if fill <= base + RESCUE_FILL and halo_fill(bw, b) <= HALO_ONLY:
                    continue
                if words is not None and not text_right_of(b, words):
                    continue
                found.append(b)
        found += _rescue_ocr_glyphs(bw, col, boxes + found, words)
    return found


# OCR hay đọc ô vuông mờ thành các ký tự này (Data2.1 trang 3: ô khớp mẫu 0.57 bị đọc thành "()").
# Không có "(|", "C)": đó là ô viền mảnh (--mixed) mà khi đã tick thì vẫn không tìm lại được -> sót dấu trong regress
CHECKBOX_OCR_GLYPHS = {")", "(", "[]", "()", "[ ]", "CO", "J", "—", "–", "-", "©", "O)", "(C", "DC", "OC",
                       "o", "O", "0", "u", "U"}


def _rescue_ocr_glyphs(bw, col, known, words):
    """Cứu ô mà OCR đọc thành ký tự kiểu "()", "J" nằm thẳng cột ô đã biết và có nhãn bên phải."""
    if not words:
        return []
    H = bw.shape[0]
    col_x = int(np.median([b["x"] for b in col]))
    # cỡ ô lấy theo các ô cùng cột (trang trộn ô to/nhỏ), không mặc định mẫu đầu tiên
    col_w, col_h = int(np.median([b["w"] for b in col])), int(np.median([b["h"] for b in col]))
    col_tpl = max(set(b["tpl"] for b in col), key=[b["tpl"] for b in col].count)
    out = []
    for w in words:
        if w["text"].strip() not in CHECKBOX_OCR_GLYPHS or abs(w["x"] - col_x) > 25 or w["w"] > 55 or w["h"] > 55:
            continue
        if w["y"] < 0.04 * H or w["y"] > 0.93 * H:
            continue
        if any(abs(o["y"] - w["y"]) < 0.8 * o["h"] for o in known + out):
            continue
        b = {"x": col_x, "y": w["y"], "w": col_w, "h": col_h, "tpl": col_tpl, "match": 0.50, "rescued": True}
        if text_right_of(b, words):
            b["fill"] = ink_fill(bw, col_x, w["y"], col_w, col_h)
            out.append(b)
    return out


def scan_boxes(bw, words, tpls, low=TM_FRAME):
    if tpls:
        boxes = find_boxes_by_template(bw, tpls, words=words, low=low)
    else:
        # không học được mẫu (trang ít ô): bỏ ô không có nhãn bên phải (nét chữ ký, con dấu - Data2.2)
        boxes = [{"x": x, "y": y, "w": w, "h": h, "fill": f, "tpl": 0}
                 for (x, y, w, h, f) in find_checkboxes(bw, words=words)]
        if words is not None:
            boxes = [b for b in boxes if text_right_of(b, words)]
    if not boxes:
        return boxes, 0.0
    for b in boxes:
        b["halo"] = halo_fill(bw, b)
    allf = float(np.median([b["fill"] for b in boxes]))
    allh = float(np.median([b["halo"] for b in boxes]))
    for b in boxes:   # mức "ô trống" tính riêng cho từng kiểu ô (ô nhỏ/mảnh có mực khác ô to)
        grp = [o for o in boxes if o["tpl"] == b["tpl"]]
        # nhóm nhỏ (3-4 ô) mà đa số đã tick thì trung vị nhóm là mức "có dấu" -> chặn trên bằng trung vị cả trang
        base = min(float(np.median([o["fill"] for o in grp])), allf) if len(grp) >= 3 else allf
        hbase = min(float(np.median([o["halo"] for o in grp])), allh) if len(grp) >= 3 else allh
        b["marked"] = (b["fill"] > base + 0.12 or (b["halo"] > hbase + 0.035 and b["fill"] > base + 0.05)
                       or (b["halo"] > hbase + HALO_ONLY and b.get("column", 3) >= 3))
        b["text"] = text_right_of(b, words)
    return boxes, allf


def scan_tables(bw, words, n_score):
    out = []
    for ti, t in enumerate(find_tables(bw)):
        rows, cols = t["rows"], t["cols"]
        ncol = len(cols) - 1
        if ncol <= n_score or len(rows) < 2:
            continue
        first = ncol - n_score
        raw_header = [text_in(words, cols[c], rows[0], cols[c + 1], rows[1]) for c in range(first, ncol)]
        is_num = sum(h.strip() in (str(i + 1), f"({i+1})") for i, h in enumerate(raw_header)) >= 2
        is_word = any(w in " ".join(raw_header).lower() for w in ["poor", "fair", "satisfactory", "good", "excellent"])
        has_header = (is_num or is_word) and len(rows) >= 3

        if has_header:
            if is_num or sum(h.strip() == str(i + 1) for i, h in enumerate(raw_header)) > len(raw_header) / 2:
                header = [str(i + 1) for i in range(len(raw_header))]
            else:
                header = raw_header
            start_r = 1
        else:
            header = [str(i + 1) for i in range(n_score)]
            start_r = 0

        data = []
        for r in range(start_r, len(rows) - 1):
            inks = [center_ink(bw, cols[c], rows[r], cols[c + 1], rows[r + 1])
                    for c in range(first, ncol)]
            base = float(np.median(inks))
            marked = [i for i, v in enumerate(inks) if v - base > CELL_DELTA]
            if not marked and len(inks) >= 2:
                top, second = sorted(inks)[-1], sorted(inks)[-2]
                faint = top - base > CELL_FAINT or (second - base <= CELL_CLEAN and top - base > CELL_CLEAN_MIN)
                if faint and top - second >= CELL_LEAD * (top - base):
                    marked = [inks.index(top)]
            row_idx = r if start_r == 1 else (r + 1)
            data.append({"row": row_idx,
                         # đọc từng cột nhãn riêng (cột STT "8." + cột nội dung 2 dòng) rồi ghép lại
                         "label": " ".join(s for s in (text_in(words, cols[c], rows[r], cols[c + 1], rows[r + 1])
                                                       for c in range(first)) if s),
                         "inks": [round(v, 3) for v in inks], "marked_cols": marked,
                         "cells": [(cols[c], rows[r], cols[c + 1], rows[r + 1])
                                   for c in range(first, ncol)]})
        out.append({"index": ti + 1, "header": header, "rows": data, "bbox": (t["x"], t["y"], t["w"], t["h"])})
    return out


def learn_template(images, words_list=None):
    return build_template([binarize(im) for im in images], words_list=words_list)


def scan_page(img, n_score, tpl=None, bw=None, text=None, words=None):
    if bw is None:
        bw = binarize(img)
    if text is None or words is None:
        text, words = ocr(img, bw)
    soft = page_is_soft(img, bw)
    boxes, base = scan_boxes(bw, words, tpl, low=TM_LOW if soft else TM_FRAME)
    tables = scan_tables(bw, words, n_score)
    # ô vuông nằm trong bảng điểm đã được tính ở phần TABLE -> bỏ khỏi danh sách CHECKBOX để không báo trùng
    def in_table(b):
        cx, cy = b["x"] + b["w"] / 2, b["y"] + b["h"] / 2
        return any(tx <= cx <= tx + tw and ty <= cy <= ty + th for tx, ty, tw, th in (t["bbox"] for t in tables))
    boxes = [b for b in boxes if not in_table(b)]
    return {"text": text, "words": words, "boxes": boxes, "box_base": base, "tables": tables, "soft": soft}


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
    ap.add_argument("--styles", default="cross,tick,circle",
                    help="kiểu dấu mô phỏng, chọn trong cross,tick,circle,fill,dash (mặc định giữ như regress)")
    ap.add_argument("--light", type=float, default=0.0, help="xác suất dấu mô phỏng vẽ bằng mực nhạt")
    ap.add_argument("--out", default="output", help="thư mục ghi ảnh *_scan.png và *_scan.json")
    ap.add_argument("--crops", help="xuất ảnh cắt quanh ô kèm nhãn vào thư mục này (export_crops.py)")
    args = ap.parse_args()
    styles = args.styles.split(",")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(args.pdf).stem
    images = list(render_pages(args.pdf))
    rng = random.Random(args.seed)
    truth = []  # (trang, loại, ...) những ô đã vẽ dấu giả

    if args.simulate and args.mixed:
        mrng = random.Random(args.seed + 100)
        for img in images:   # truyền từ OCR để không vẽ lại chữ "O" thành ô vuông
            bw0 = binarize(img)
            for (x, y, w, h, _) in find_checkboxes(bw0, words=ocr(img, bw0)[1]):
                r = mrng.random()
                if r < 0.5:
                    cv2.rectangle(img, (x - 3, y - 3), (x + w + 3, y + h + 3), (255, 255, 255), -1)
                    n = int(w * 0.65) if r < 0.25 else w   # ô nhỏ, hoặc cùng cỡ nhưng viền mảnh
                    x0, y0 = x + (w - n) // 2, y + (h - n) // 2
                    cv2.rectangle(img, (x0, y0), (x0 + n, y0 + n), (0, 0, 0), 2 if r < 0.25 else 1)

    def ink(rng):   # --light = 0 thì không gọi rng -> chuỗi ngẫu nhiên giữ nguyên như baseline regress
        if args.light and rng.random() < args.light:
            v = rng.randint(110, 170)
            return (v, v, v)
        return (0, 0, 0)

    if args.simulate:
        sim_words_list = [ocr(im, binarize(im))[1] for im in images]
        tpl0 = learn_template(images, words_list=sim_words_list)
        for pno, (img, w_list) in enumerate(zip(images, sim_words_list), 1):
            page = scan_page(img, args.score_cols, tpl0, words=w_list)
            for bi, b in enumerate(page["boxes"]):
                if rng.random() < 0.3:
                    k = 1.7 if rng.random() < args.overshoot else 0.5   # 1.7 -> nét vẽ tràn ra ngoài ô
                    cx, cy = b["x"] + b["w"] / 2, b["y"] + b["h"] / 2
                    box = (int(cx - k * b["w"]), int(cy - k * b["h"]), int(cx + k * b["w"]), int(cy + k * b["h"]))
                    draw_mark(img, box, rng.choice(styles), rng, ink(rng))
                    truth.append((pno, "box", (b["x"], b["y"], b["w"], b["h"])))
            for t in page["tables"]:
                for r in t["rows"]:
                    c = rng.randrange(len(r["cells"]))
                    draw_mark(img, r["cells"][c], rng.choice(styles), rng, ink(rng))
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

    # Chạy OCR một lần cho từng trang để lấy danh sách từ hỗ trợ loại bỏ false positive khi học template
    pages_ocr = []
    for im in images:
        bw = binarize(im)
        txt, words = ocr(im, bw)
        pages_ocr.append((bw, txt, words))

    words_list = [p[2] for p in pages_ocr]
    results = []
    tpl = learn_template(images, words_list=words_list)  # học hình ô trống từ chính tài liệu đang quét
    if tpl:
        print("Mẫu ô vuông học được:", ", ".join(f"{t['w']}x{t['h']} px ({t['n']} ô)" for t in tpl))
    for pno, (img, (bw, text, words)) in enumerate(zip(images, pages_ocr), 1):
        page = scan_page(img, args.score_cols, tpl, bw=bw, text=text, words=words)
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

    if args.crops:
        from export_crops import export
        if args.simulate:
            export(images, results, args.crops, f"{stem}_sim{args.seed}", drawn=truth)
        else:
            export(images, results, args.crops, stem,
                   truth_path=TRUTH / f"{stem}.json")


if __name__ == "__main__":
    main()