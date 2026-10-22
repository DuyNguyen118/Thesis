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


def _corner_angles(approx):
    """Tính góc tại mỗi đỉnh của polygon; ô vuông có 4 góc gần 90°, chữ tròn thì không."""
    pts = approx.reshape(-1, 2).astype(float)
    n = len(pts)
    angles = []
    for i in range(n):
        v1 = pts[(i - 1) % n] - pts[i]
        v2 = pts[(i + 1) % n] - pts[i]
        cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
        angles.append(np.degrees(np.arccos(np.clip(cos, -1, 1))))
    return angles


def _is_rectangular(approx, tol=35):
    """Polygon có ít nhất 4 góc gần 90° (±tol) — loại chữ tròn O, D, C..."""
    angles = _corner_angles(approx)
    return sum(abs(a - 90) < tol for a in angles) >= 4


def _border_density(bw, x, y, w, h):
    """Mật độ mực ở viền ô (dải sát 4 cạnh), để xác nhận có khung bao quanh."""
    b = max(2, int(0.15 * min(w, h)))  # độ dày dải viền
    top = bw[y:y + b, x:x + w]
    bot = bw[y + h - b:y + h, x:x + w]
    left = bw[y:y + h, x:x + b]
    right = bw[y:y + h, x + w - b:x + w]
    total = sum(s.size for s in [top, bot, left, right])
    if total == 0:
        return 0.0
    ink = sum(float((s > 0).sum()) for s in [top, bot, left, right])
    return ink / total


def find_checkboxes(bw, dpi=DPI, words=None):
    """Tìm các ô checkbox sạch (chưa đánh dấu hoặc viền rõ ràng).

    Hỗ trợ cả ô vuông chuẩn (~1:1, M02) và ô chữ nhật Unicode font (~0.6:1, CamScanner).
    Kích thước chuẩn ở 300 DPI: w trong khoảng 16-55px, h trong khoảng 20-55px.
    Nếu truyền `words` (danh sách từ từ OCR), tự động lọc bỏ các chữ cái nằm trong từ.
    """
    H, W = bw.shape
    lo, hi = int(1.5 * dpi / 25.4), int(5.5 * dpi / 25.4)

    multi_words = []
    if words:
        for w_obj in words:
            txt = w_obj.get("text", "").strip()
            if len(txt) >= 2:
                multi_words.append(w_obj)

    contours, hier = cv2.findContours(bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if hier is None:
        return []
    hier0 = hier[0]

    boxes = []
    seen = set()

    for i, c in enumerate(contours):
        x, y, w, h = cv2.boundingRect(c)

        # 1. Kích thước (khoảng 1.5 - 5.5 mm)
        if not (lo <= w <= hi and lo <= h <= hi):
            continue

        # 2. Tỉ lệ khung hình: cho phép từ 0.50 (chữ nhật đứng) đến 1.30 (vuông hoặc hơi ngang)
        ratio = w / h
        if not (0.50 <= ratio <= 1.30):
            continue

        # 3. Bỏ qua vùng lề trang (tiêu đề đầu trang, chân trang CamScanner, viền mép)
        if x < 0.05 * W or x > 0.88 * W or y < 0.04 * H or y > 0.94 * H:
            continue

        cx, cy = x + w / 2, y + h / 2

        # 4. Loại bỏ các chữ cái nằm bên trong một từ OCR hoàn chỉnh (ví dụ chữ O trong EVALUATION)
        in_word = False
        for mw in multi_words:
            if mw["w"] > w * 1.3:
                if mw["x"] + 4 <= cx <= mw["x"] + mw["w"] - 4 and mw["y"] - 3 <= cy <= mw["y"] + mw["h"] + 3:
                    in_word = True
                    break
        if in_word:
            continue

        # 5. Kiểm tra ruột rỗng (hollow): ô chưa đánh dấu phải có ruột tương đối rỗng
        m = max(3, int(0.20 * min(w, h)))
        inner = bw[y + m : y + h - m, x + m : x + w - m]
        inner_fill = float((inner > 0).mean()) if inner.size else 1.0

        child = hier0[i][2]
        has_hole = (child != -1)

        # Cần có lỗ contour con hoặc mật độ mực vùng ruột rất thấp (< 25%)
        if not (has_hole or inner_fill < 0.22):
            continue
        if inner_fill > 0.30:
            continue

        # 6. Kiểm tra viền: ít nhất 3 cạnh phải có nét mực bao quanh
        b = max(2, int(0.15 * min(w, h)))
        top = bw[y : y + b, x : x + w]
        bot = bw[y + h - b : y + h, x : x + w]
        left = bw[y : y + h, x : x + b]
        right = bw[y : y + h, x + w - b : x + w]
        sides_ink = [
            float((top > 0).mean()) if top.size else 0,
            float((bot > 0).mean()) if bot.size else 0,
            float((left > 0).mean()) if left.size else 0,
            float((right > 0).mean()) if right.size else 0,
        ]
        if sum(s >= 0.25 for s in sides_ink) < 3:
            continue

        # Tránh trùng lặp
        duplicate = False
        for sx, sy in seen:
            if abs(sx - x) < 0.5 * w and abs(sy - y) < 0.5 * h:
                duplicate = True
                break
        if duplicate:
            continue

        seen.add((x, y))
        boxes.append((x, y, w, h, inner_fill))

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