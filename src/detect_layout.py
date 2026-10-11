import sys
from pathlib import Path

import cv2
import numpy as np
import pymupdf

DPI = 300


def render_pages(pdf_path, dpi=DPI):
    doc = pymupdf.open(pdf_path)
    for page in doc:
        pix = page.get_pixmap(dpi=dpi)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if pix.n == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        elif pix.n == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        yield img


def binarize(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Otsu + đảo màu: nét đen thành trắng để dùng morphology
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return bw


def find_checkboxes(bw, dpi=DPI):
    """Ô vuông nhỏ: viền chữ nhật rỗng, tỷ lệ ~1:1, kích thước khoảng 2-5 mm."""
    mm = dpi / 25.4
    lo, hi = 2.0 * mm, 5.5 * mm
    contours, hier = cv2.findContours(bw, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    if hier is None:
        return boxes
    hier = hier[0]
    for i, c in enumerate(contours):
        if hier[i][3] != -1:          # bỏ contour là lỗ, chỉ xét viền ngoài
            continue
        x, y, w, h = cv2.boundingRect(c)
        if not (lo <= w <= hi and lo <= h <= hi):
            continue
        if abs(w - h) > 0.15 * max(w, h):
            continue
        # viền ngoài phải là hình chữ nhật thật (4 đỉnh), chữ O/D/A... thì không
        approx = cv2.approxPolyDP(c, 0.04 * cv2.arcLength(c, True), True)
        if len(approx) != 4:
            continue
        # phải có lỗ bên trong chiếm gần hết ô (viền mỏng)
        child = hier[i][2]
        if child == -1:
            continue
        hole_area = cv2.contourArea(contours[child])
        if hole_area < 0.55 * (w * h):
            continue
        inner = bw[y + h // 4: y + 3 * h // 4, x + w // 4: x + 3 * w // 4]
        fill = float(inner.mean()) / 255.0
        boxes.append((x, y, w, h, fill))
    boxes.sort(key=lambda b: (b[1], b[0]))
    return boxes


def find_table_lines(bw, dpi=DPI):
    """Tìm đường kẻ ngang/dọc dài bằng morphology."""
    h, w = bw.shape
    min_len_h = int(w * 0.08)
    min_len_v = int(dpi / 25.4 * 8)
    horiz = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (min_len_h, 1)))
    vert = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, min_len_v)))
    return horiz, vert


def group_positions(mask, axis):
    """Gom các đường liền kề thành 1 vị trí (trung bình)."""
    proj = mask.sum(axis=axis) > 0
    pos, start = [], None
    for i, v in enumerate(proj):
        if v and start is None:
            start = i
        elif not v and start is not None:
            pos.append((start + i - 1) // 2)
            start = None
    if start is not None:
        pos.append((start + len(proj) - 1) // 2)
    return pos


def main(pdf_path):
    out = Path("output")
    out.mkdir(exist_ok=True)
    stem = Path(pdf_path).stem
    for i, img in enumerate(render_pages(pdf_path), 1):
        bw = binarize(img)
        boxes = find_checkboxes(bw)
        horiz, vert = find_table_lines(bw)
        rows = group_positions(horiz, axis=1)
        cols = group_positions(vert, axis=0)

        vis = img.copy()
        for (x, y, w, h, fill) in boxes:
            color = (0, 0, 255) if fill > 0.25 else (0, 160, 0)  # đỏ = có dấu, xanh = trống
            cv2.rectangle(vis, (x - 3, y - 3), (x + w + 3, y + h + 3), color, 3)
        for y in rows:
            cv2.line(vis, (0, y), (vis.shape[1], y), (255, 0, 0), 1)
        for x in cols:
            cv2.line(vis, (x, 0), (x, vis.shape[0]), (255, 0, 0), 1)

        path = out / f"{stem}_p{i}.png"
        cv2.imwrite(str(path), vis)
        print(f"Trang {i}: {img.shape[1]}x{img.shape[0]} px | checkbox: {len(boxes)} | "
              f"đường ngang: {len(rows)} | đường dọc: {len(cols)} -> {path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Cách dùng: python src/detect_layout.py <file.pdf>")
    main(sys.argv[1])