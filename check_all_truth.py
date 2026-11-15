import json
import sys
sys.path.append("src")
from common import TRUTH

for f in sorted(TRUTH.glob("*.json")):
    with open(f, encoding="utf-8") as fp:
        d = json.load(fp)
    n_boxes = sum(len(p.get("boxes", [])) for p in d.get("pages", []))
    n_m_boxes = sum(len([b for b in p.get("boxes", []) if b.get("marked")]) for p in d.get("pages", []))
    n_rows = sum(len(r) for p in d.get("pages", []) for t in p.get("tables", []) for r in [t.get("rows", [])])
    n_m_rows = sum(len([r for r in t.get("rows", []) if r.get("marked_cols")]) for p in d.get("pages", []) for t in p.get("tables", []))
    print(f"{f.stem:<30} verified: {d.get('verified')!s:<6} | boxes: {n_boxes:<3} (marked: {n_m_boxes:<2}) | table rows: {n_rows:<2} (marked: {n_m_rows:<2})")

