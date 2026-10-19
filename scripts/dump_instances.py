"""Сохранение экземпляров задач, использованных в exp_solvers.py и
exp_lps_verify.py (регенерация по тем же зёрнам с проверкой)."""
import csv, gzip, json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from generators import random_dnf, tiered_kb, general_kb
from lps import lit_str, odv_bitsets, consistent

def lps_json(S):
    return dict(terminals=S.terminals, prods=[[[lit_str(b) for b in L], lit_str(a)] for L, a in S.prods])

rows = {r["seed"]: r for r in csv.DictReader(open(ROOT / "results/solvers_rows.csv"))}
with gzip.open(ROOT / "data/solvers_DNF.jsonl.gz", "wt") as f:
    for big, n in enumerate((100, 1000)):
        for im, m in enumerate((100, 1000, 10000)):
            for k in range(5):
                seed = 90000 + 1000 * big + 10 * im + k
                rng = np.random.default_rng(seed)
                D = random_dnf(rng, n, m, rank=(5, 20))
                c = [int(v) for v in rng.integers(-100, 101, size=n)]
                f.write(json.dumps(dict(seed=seed, n=n, D=D, c=c, opt=int(rows[str(seed)]["ours_val"]))) + "\n")
with gzip.open(ROOT / "data/solvers_KB.jsonl.gz", "wt") as f:
    for a, n in enumerate((50, 100, 200)):
        for b, H in enumerate((4, 6, 8)):
            for k in range(5):
                seed = 95000 + 100 * a + 10 * b + k
                rng = np.random.default_rng(seed)
                S = tiered_kb(rng, n, H, facts=(3, 6))
                c = [int(v) for v in rng.integers(-100, 101, size=n)]
                f.write(json.dumps(dict(seed=seed, n=n, H=H, c=c, opt=int(rows[str(seed)]["HiGHS_val"]), **lps_json(S))) + "\n")
with gzip.open(ROOT / "data/solvers_KB-large.jsonl.gz", "wt") as f:
    for n, H, fc in ((1000, 10, (20, 40)), (5000, 15, (100, 200)), (20000, 20, (400, 800))):
        seed = 97000 + n
        rng = np.random.default_rng(seed)
        S = tiered_kb(rng, n, H, facts=fc)
        c = [int(v) for v in rng.integers(-100, 101, size=n)]
        f.write(json.dumps(dict(seed=seed, n=n, H=H, c=c, **lps_json(S))) + "\n")
cnt, seed = 0, 0
with gzip.open(ROOT / "data/lps_verify_G2.jsonl.gz", "wt") as f:
    while cnt < 3000:
        rng = np.random.default_rng(seed)
        S = general_kb(rng)
        v = odv_bitsets(S)
        if ("y", 1) in v and v[("y", 1)].any() and consistent(S):
            cnt += 1
            f.write(json.dumps(dict(seed=seed, **lps_json(S))) + "\n")
        seed += 1
print("G2 seeds tried:", seed)
