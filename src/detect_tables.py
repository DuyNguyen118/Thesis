"""Tách bảng (Attendance/Punctuality của M02, bảng điểm 1-5 của M03) và vẽ dấu giả cho --simulate.

Đọc dấu trong ô bảng nằm ở scan_form.scan_tables.
"""
import cv2
import numpy as np

from detect_layout import DPI, find_table_lines, group_positions


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
    lines = cv2.dilate(cv2.bitwise_or(horiz, vert), np.ones((7, 7), np.uint8))
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
        # Bổ sung cột nếu đường ngang kéo dài vượt ra ngoài đường kẻ dọc ngoài cùng
        # (trường hợp bảng bị scan cắt mất nét viền hoặc đứt viền ngoài cùng như Data9.1)
        if len(cols) >= 2:
            hz_cols = np.where((hz > 0).any(axis=0))[0] + x
            if len(hz_cols):
                hz_x0, hz_x1 = int(hz_cols[0]), int(hz_cols[-1])
                med_w = float(np.median(np.diff(cols)))
                if med_w > 10 * mm:
                    extra_r = int(round((hz_x1 - cols[-1]) / med_w))
                    for _ in range(extra_r):
                        cols.append(int(round(cols[-1] + med_w)))
                    extra_l = int(round((cols[0] - hz_x0) / med_w))
                    for _ in range(extra_l):
                        cols.insert(0, int(round(cols[0] - med_w)))
        tables.append({"x": int(x), "y": int(y), "w": int(w), "h": int(h), "rows": rows, "cols": cols})
    tables.sort(key=lambda t: t["y"])
    return tables


def draw_mark(img, box, style, rng, ink=(0, 0, 0)):
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    cx = (x0 + x1) // 2 + rng.randint(-w // 10, w // 10)
    cy = (y0 + y1) // 2 + rng.randint(-h // 10, h // 10)
    r = int(min(w, h) * 0.28)
    t = 4
    if style == "cross":
        cv2.line(img, (cx - r, cy - r), (cx + r, cy + r), ink, t)
        cv2.line(img, (cx - r, cy + r), (cx + r, cy - r), ink, t)
    elif style == "tick":
        pts = np.array([[cx - r, cy], [cx - r // 3, cy + r], [cx + r, cy - r]], np.int32)
        cv2.polylines(img, [pts], False, ink, t)
    elif style == "fill":   # tô kín bằng nét ngoằn ngoèo dày đặc
        pts = np.array([[x, cy - r if i % 2 else cy + r] for i, x in enumerate(range(cx - r, cx + r + 1, t))], np.int32)
        cv2.polylines(img, [pts], False, ink, t)
    elif style == "dash":   # gạch ngang
        cv2.line(img, (cx - r, cy), (cx + r, cy), ink, t)
    else:
        cv2.circle(img, (cx, cy), r, ink, t)