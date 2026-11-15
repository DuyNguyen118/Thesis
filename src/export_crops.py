"""Cắt ảnh quanh từng ô (và từng ô bảng) kèm nhãn để train bộ phân loại dấu (task.md 2B.1).

Không chạy riêng: được gọi từ scan_form.py khi có cờ --crops, dùng đúng ảnh đã deskew của lần quét đó.
Nhãn lấy từ:
  - file thật: data/truth/<stem>.json (ghép ô theo vị trí như evaluate.py, bảng theo IoU)
  - --simulate: danh sách dấu đã vẽ giả

Mỗi tài liệu ghi vào một thư mục riêng (để chia train/test theo FILE):
    <crops>/<doc>/<nhãn>/<doc>_p<trang>_<kiểu><i>.png     ảnh xám CROP_SIZE x CROP_SIZE
    <crops>/<doc>/labels.csv                             một dòng / ảnh, kèm kết quả của luật hiện tại
Nhãn: marked (có dấu), empty (trống), not_box (máy tìm ra nhưng đáp án không có ô ở đó).

Chạy (qua scan_form.py):
    python src/scan_form.py data/samples/Data1.1.pdf --crops data/crops
    python src/scan_form.py data/samples/M02_Intern-evaluation-template.pdf --simulate --seed 7 --crops data/crops
Sinh cả bộ dữ liệu: python src/make_dataset.py
"""
import csv
import json
import shutil
from pathlib import Path

import cv2

from evaluate import _iou, _match_boxes

CROP_SIZE = 64
BOX_SCALE = 2.4      # cạnh ảnh cắt = 2.4 * cạnh lớn của ô -> thấy được dấu tràn/khoanh ngoài ô (như vùng halo)
CELL_MARGIN = 4      # cắt ô bảng rộng thêm bấy nhiêu px mỗi phía
FIELDS = ["file", "doc", "source", "page", "kind", "x", "y", "w", "h", "label",
          "detected", "rule_marked", "fill", "halo", "match"]


def _square(gray, x0, y0, x1, y1, cut_right_at=None):
    """Cắt vùng (x0, y0)-(x1, y1), đệm trắng thành hình vuông (cả phần ra ngoài trang), resize CROP_SIZE."""
    side = max(x1 - x0, y1 - y0)
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    sx, sy = cx - side // 2, cy - side // 2
    pad = cv2.copyMakeBorder(gray, side, side, side, side, cv2.BORDER_CONSTANT, value=255)
    roi = pad[sy + side:sy + 2 * side, sx + side:sx + 2 * side].copy()
    if cut_right_at is not None:
        c = cut_right_at - sx
        if 0 <= c < side:
            roi[:, c:] = 255
    return cv2.resize(roi, (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_AREA)


def crop_box(gray, x, y, w, h):
    s = int(round(BOX_SCALE * max(w, h)))
    cx, cy = x + w // 2, y + h // 2
    g = max(4, int(0.15 * w))
    return _square(gray, cx - s // 2, cy - s // 2, cx + s // 2, cy + s // 2, cut_right_at=x + w + g)


def crop_cell(gray, cell):
    x0, y0, x1, y1 = cell
    return _square(gray, x0 - CELL_MARGIN, y0 - CELL_MARGIN, x1 + CELL_MARGIN, y1 + CELL_MARGIN)


def _rule(b):
    """Cột kết quả của luật hiện tại (baseline) cho ô máy tìm ra."""
    return {"detected": 1, "rule_marked": int(bool(b["marked"])), "fill": round(b["fill"], 4),
            "halo": round(b["halo"], 4), "match": b.get("match", "")}


MISSED = {"detected": 0, "rule_marked": 0, "fill": "", "halo": "", "match": ""}


def _cells(page, marked_of):
    """Mẫu cho ô bảng. `marked_of(t, row)` -> danh sách cột có dấu theo đáp án, hoặc None nếu không biết."""
    out = []
    for t in page["tables"]:
        for r in t["rows"]:
            cols = marked_of(t, r["row"])
            if cols is None:
                continue
            for c, cell in enumerate(r["cells"]):
                x0, y0, x1, y1 = cell
                out.append({"kind": "cell", "x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0, "cell": cell,
                            "label": "marked" if c in cols else "empty",
                            "detected": 1, "rule_marked": int(c in r["marked_cols"]),
                            "fill": r["inks"][c], "halo": "", "match": ""})
    return out


def label_real(results, truth):
    """Ghép kết quả quét với data/truth/<stem>.json -> danh sách mẫu theo trang."""
    tpages = {p["page"]: p for p in truth["pages"]}
    samples = []
    for pno, page in enumerate(results, 1):
        tp = tpages.get(pno, {"boxes": [], "tables": []})
        pairs = dict((j, i) for i, j in _match_boxes(page["boxes"], tp["boxes"]))
        for j, tb in enumerate(tp["boxes"]):   # ô đáp án: cắt theo tọa độ đáp án, kể cả khi máy không tìm ra
            i = pairs.get(j)
            samples.append({"page": pno, "kind": "box", "x": tb["x"], "y": tb["y"], "w": tb["w"], "h": tb["h"],
                            "label": "marked" if tb["marked"] else "empty",
                            **(_rule(page["boxes"][i]) if i is not None else MISSED)})
        for i, b in enumerate(page["boxes"]):
            if i not in pairs.values():
                samples.append({"page": pno, "kind": "box", "x": b["x"], "y": b["y"], "w": b["w"], "h": b["h"],
                                "label": "not_box", **_rule(b)})

        def marked_of(t, row):
            tt = max(tp["tables"], key=lambda x: _iou([int(v) for v in t["bbox"]], x["bbox"]), default=None)
            if tt is None or _iou([int(v) for v in t["bbox"]], tt["bbox"]) < 0.5:
                return None
            tr = next((x for x in tt["rows"] if x["row"] == row), None)
            return tr["marked_cols"] if tr else None
        samples += [{"page": pno, **s} for s in _cells(page, marked_of)]
    return samples


def label_sim(results, drawn):
    """`drawn` là danh sách dấu đã vẽ của --simulate: (trang, "box", (x, y, w, h)) / (trang, "cell", (bảng, hàng, cột))."""
    samples = []
    for pno, page in enumerate(results, 1):
        boxes = [k for p, kind, k in drawn if p == pno and kind == "box"]
        hit = set()
        for b in page["boxes"]:
            k = next((k for k in boxes if abs(b["x"] - k[0]) < 0.5 * b["w"] and abs(b["y"] - k[1]) < 0.5 * b["h"]), None)
            if k is not None:
                hit.add(k)
            samples.append({"page": pno, "kind": "box", "x": b["x"], "y": b["y"], "w": b["w"], "h": b["h"],
                            "label": "marked" if k is not None else "empty", **_rule(b)})
        for k in boxes:   # dấu đã vẽ mà máy không tìm ra ô -> vẫn là mẫu "có dấu" khó, cắt theo tọa độ lúc vẽ
            if k not in hit:
                samples.append({"page": pno, "kind": "box", "x": k[0], "y": k[1], "w": k[2], "h": k[3],
                                "label": "marked", **MISSED})
        cells = {k for p, kind, k in drawn if p == pno and kind == "cell"}
        samples += [{"page": pno, **s} for s in _cells(
            page, lambda t, row: [c for (ti, r, c) in cells if ti == t["index"] and r == row])]
    return samples


def export(images, results, out_dir, doc, truth_path=None, drawn=None):
    """Ghi ảnh cắt + labels.csv của một tài liệu. Trả về số ảnh theo nhãn."""
    if drawn is not None:
        samples, source = label_sim(results, drawn), "sim"
    elif truth_path and Path(truth_path).exists():
        with open(truth_path, encoding="utf-8") as f:
            truth = json.load(f)
        if not truth.get("verified"):
            print(f"CẢNH BÁO: {truth_path} chưa verified, nhãn ảnh cắt có thể sai")
        samples, source = label_real(results, truth), "real"
    else:
        print(f"Không có đáp án {truth_path} -> không xuất ảnh cắt (tạo bằng make_truth.py trước)")
        return {}

    d = Path(out_dir) / doc
    shutil.rmtree(d, ignore_errors=True)   # chạy lại cùng tài liệu: xóa ảnh cũ để không lẫn với labels.csv mới
    d.mkdir(parents=True)
    grays = [cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) for im in images]
    rows, count = [], {}
    for i, s in enumerate(samples):
        g = grays[s["page"] - 1]
        img = crop_cell(g, s.pop("cell")) if s["kind"] == "cell" else crop_box(g, s["x"], s["y"], s["w"], s["h"])
        name = f"{s['label']}/{doc}_p{s['page']}_{s['kind']}{i}.png"
        (d / s["label"]).mkdir(exist_ok=True)
        cv2.imwrite(str(d / name), img)
        rows.append({"file": name, "doc": doc, "source": source, **s})
        count[s["label"]] = count.get(s["label"], 0) + 1
    with open(d / "labels.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"Ảnh cắt: {d} " + ", ".join(f"{k} {v}" for k, v in sorted(count.items())))
    return count
