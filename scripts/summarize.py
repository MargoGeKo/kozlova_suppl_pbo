"""Сводная табл. 7 статьи по results/solvers_rows.csv; D_size -- |D|."""
import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
rows = list(csv.DictReader(open(ROOT / "results/solvers_rows.csv")))
methods = ["ours"] + [k[:-4] for k in rows[0] if k.endswith("_val") and k != "ours_val"]
out = []
mismatch = 0
for fam, keys in (("DNF", ("n", "m")), ("KB", ("n", "H"))):
    cells = sorted({tuple(int(r[k]) for k in keys) for r in rows if r["family"] == fam})
    for cell in cells:
        R = [r for r in rows if r["family"] == fam and tuple(int(r[k]) for k in keys) == cell]
        rec = dict(family=fam, **dict(zip(keys, cell)), tasks=len(R),
                   size=int(np.median([int(r["vars"]) for r in R])),
                   D_size=("%g" % np.median([float(r["D_size"]) for r in R if r["D_size"]])) if any(r["D_size"] for r in R) else "-")
        for mth in methods:
            solved = [r for r in R if r[mth + "_t"] not in ("", "None") and float(r[mth + "_t"]) < 60
                      and (r[mth + "_val"] not in ("", "None") or mth == "ours")]
            ts = [float(r[mth + "_t"]) if r in solved else 60.0 for r in R]
            rec[mth + "_solved"] = len(solved)
            rec[mth + "_med_t"] = round(float(np.median(ts)), 4)
        for r in R:
            vals = {r[m + "_val"] for m in methods if r[m + "_val"] not in ("", "None")}
            mismatch += len(vals) > 1
        out.append(rec)
with open(ROOT / "results/summary_solvers.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in out for k in r)), restval=""); w.writeheader(); w.writerows(out)
for r in out:
    print(r)
print("value mismatches:", mismatch)
