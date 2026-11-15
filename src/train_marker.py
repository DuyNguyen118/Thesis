"""Train CNN nhỏ phân loại ảnh cắt ô: có dấu / trống (task.md 2B.3-2B.4). Chạy CPU được (laptop 8GB RAM).

Dữ liệu: data/crops/*/labels.csv (sinh bằng make_dataset.py). Chia theo TÀI LIỆU, không theo ảnh:
  - train / val: các tài liệu mô phỏng (val = VAL_FRAC số tài liệu, rút theo seed)
  - test: mọi tài liệu thật (source = real) -> không bao giờ dùng để train
Nhãn not_box hiện gộp vào "trống" (mô hình 2 lớp: marked vs còn lại).

Cuối cùng in trên tập test: CNN vs luật hiện tại (cột rule_marked của labels.csv), và liệt kê ảnh CNN đoán sai.
Lưu ý: luật đã được chỉnh tay trên chính Data1.1/Data1.2 nên baseline luật trên test này đang được lợi.

Ghi ra models/marker_cnn.pt (trọng số) và models/marker_metrics.json.

Cài (một lần, bản CPU):
    pip install torch --index-url https://download.pytorch.org/whl/cpu
Chạy:
    python src/train_marker.py
    python src/train_marker.py --epochs 3          # chạy thử nhanh
"""
import argparse
import csv
import json
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from common import CROPS, MODELS, ROOT

VAL_FRAC = 0.15      # tỉ lệ tài liệu mô phỏng giữ lại làm val
THRESH = 0.5         # xác suất "có dấu" >= ngưỡng -> marked
UNSURE = (0.2, 0.8)  # xác suất trong khoảng này -> tính là "cần xem lại"


def load_rows(crops):
    rows = []
    for f in sorted(Path(crops).glob("*/labels.csv")):
        with open(f, encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                r["path"] = str(f.parent / r["file"])
                r["y_true"] = int(r["label"] == "marked")
                rows.append(r)
    return rows


def split(rows, seed):
    sim_docs = sorted({r["doc"] for r in rows if r["source"] == "sim"})
    random.Random(seed).shuffle(sim_docs)
    val_docs = set(sim_docs[:max(1, int(len(sim_docs) * VAL_FRAC))])
    train = [r for r in rows if r["source"] == "sim" and r["doc"] not in val_docs]
    val = [r for r in rows if r["doc"] in val_docs]
    test = [r for r in rows if r["source"] == "real"]
    return train, val, test


def augment(img, rng):
    """Biến đổi nhẹ giống scan thật: lệch vài px, xoay nhỏ, co giãn, đổi sáng/tương phản, mờ, nhiễu."""
    h, w = img.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-4, 4), rng.uniform(0.9, 1.1))
    m[:, 2] += rng.uniform(-3, 3, size=2)
    img = cv2.warpAffine(img, m, (w, h), borderValue=255).astype(np.float32)
    img = (img - 255) * rng.uniform(0.6, 1.2) + 255 - rng.uniform(0, 40)   # mực nhạt/đậm, nền xám
    if rng.random() < 0.3:
        img = cv2.GaussianBlur(img, (3, 3), 0)
    img += rng.normal(0, rng.uniform(0, 8), img.shape)
    return np.clip(img, 0, 255)


class Crops(Dataset):
    def __init__(self, rows, train=False):
        self.rows, self.train = rows, train
        self.imgs = [cv2.imread(r["path"], cv2.IMREAD_GRAYSCALE) for r in rows]   # ~5k ảnh 64x64 = ~20MB
        self.rng = np.random.default_rng()

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        img = self.imgs[i]
        img = augment(img, self.rng) if self.train else img.astype(np.float32)
        x = torch.from_numpy(1.0 - img / 255.0).float()[None]   # mực = 1, nền = 0
        return x, self.rows[i]["y_true"]


def block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(), nn.MaxPool2d(2))


class MarkerCNN(nn.Module):
    """4 khối conv 64->32->16->8->4, pool toàn cục, 1 lớp FC. ~80k tham số."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(block(1, 16), block(16, 32), block(32, 64), block(64, 96))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(96, 2))

    def forward(self, x):
        return self.head(self.features(x))


@torch.no_grad()
def predict(model, rows):
    model.eval()
    probs = []
    for x, _ in DataLoader(Crops(rows), batch_size=256):
        probs += torch.softmax(model(x), 1)[:, 1].tolist()
    return probs


def scores(y, p):
    tp = sum(a and b for a, b in zip(y, p))
    fp = sum(b and not a for a, b in zip(y, p))
    fn = sum(a and not b for a, b in zip(y, p))
    acc = sum(a == b for a, b in zip(y, p)) / len(y) if y else 0.0
    prec = tp / (tp + fp) if tp + fp else 1.0
    rec = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"n": len(y), "acc": round(acc, 4), "precision": round(prec, 4), "recall": round(rec, 4),
            "f1": round(f1, 4), "tp": tp, "fp": fp, "fn": fn}


def fmt(name, s):
    return (f"{name:10} n={s['n']:5}  acc {s['acc']:.3f}  P {s['precision']:.3f}  R {s['recall']:.3f}  "
            f"F1 {s['f1']:.3f}  (sót {s['fn']}, báo nhầm {s['fp']})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops", default=str(CROPS))
    ap.add_argument("--out", default=str(MODELS))
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    train, val, test = split(load_rows(args.crops), args.seed)
    for name, rs in (("train", train), ("val", val), ("test", test)):
        print(f"{name:5}: {len(rs):5} ảnh, {sum(r['y_true'] for r in rs):4} có dấu, "
              f"{len({r['doc'] for r in rs})} tài liệu")

    model = MarkerCNN()
    print(f"Tham số: {sum(p.numel() for p in model.parameters()):,}")
    pos = sum(r["y_true"] for r in train)
    weight = torch.tensor([1.0, (len(train) - pos) / max(pos, 1)])   # cân bằng lớp (ô trống nhiều gấp ~3)
    loss_fn = nn.CrossEntropyLoss(weight=weight)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    loader = DataLoader(Crops(train, train=True), batch_size=args.batch, shuffle=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    best = -1.0
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        total = 0.0
        for x, y in loader:
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            total += loss.item() * len(y)
        sched.step()
        sv = scores([r["y_true"] for r in val], [p >= THRESH for p in predict(model, val)])
        print(f"epoch {ep:2}: loss {total / len(train):.4f} | val F1 {sv['f1']:.4f} acc {sv['acc']:.4f} "
              f"| {time.time() - t0:.0f}s")
        if sv["f1"] >= best:   # giữ mô hình tốt nhất theo val (mô phỏng), KHÔNG chọn theo test
            best = sv["f1"]
            torch.save(model.state_dict(), out / "marker_cnn.pt")

    model.load_state_dict(torch.load(out / "marker_cnn.pt"))
    prob = predict(model, test)
    y = [r["y_true"] for r in test]
    cnn = [p >= THRESH for p in prob]
    rule = [r["rule_marked"] == "1" for r in test]   # ô máy sót (detected = 0) thì luật coi như không dấu
    unsure = sum(UNSURE[0] < p < UNSURE[1] for p in prob)
    res = {"test": {"cnn": scores(y, cnn), "rule": scores(y, rule), "cnn_unsure": unsure},
           "val_best_f1": best, "args": vars(args)}
    for kind in ("box", "cell"):
        idx = [i for i, r in enumerate(test) if r["kind"] == kind]
        res["test"][f"cnn_{kind}"] = scores([y[i] for i in idx], [cnn[i] for i in idx])
        res["test"][f"rule_{kind}"] = scores([y[i] for i in idx], [rule[i] for i in idx])

    print("\n=== TEST (scan thật, chưa từng train) ===")
    print(fmt("CNN", res["test"]["cnn"]))
    print(fmt("Luật", res["test"]["rule"]))
    for kind in ("box", "cell"):
        print(fmt(f"  CNN {kind}", res["test"][f"cnn_{kind}"]))
        print(fmt(f"  Luật {kind}", res["test"][f"rule_{kind}"]))
    print(f"CNN không chắc ({UNSURE[0]}-{UNSURE[1]}): {unsure} ảnh")
    wrong = [(r, p) for r, p, c, t in zip(test, prob, cnn, y) if c != t]
    if wrong:
        print("\nCNN đoán sai:")
        for r, p in wrong:
            print(f"  {r['path']}  nhãn {r['label']}, P(dấu) = {p:.2f}, luật = {r['rule_marked']}")
    with open(out / "marker_metrics.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f"\nĐã lưu {out / 'marker_cnn.pt'} và marker_metrics.json")


if __name__ == "__main__":
    main()
