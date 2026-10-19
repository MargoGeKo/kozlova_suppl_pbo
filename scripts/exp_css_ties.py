"""Чувствительность стратегии текущего решения (CSS) к разрешению ничьих
в MR/PMR (группа 2 алгоритма 7). Правила: 'first' (первый по номеру,
как в exp_alg7), 'far' (противник с наибольшим числом различий),
'near' (с наименьшим), 'rand' (случайно, 5 зёрен)."""
import csv, json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from alg7 import Problem, build_classes, worst_case

def make_css(tb, rng=None):
    def strat(prob, answers):
        po, groups = prob.PO(answers)
        cand = []
        for a in po:
            vals = [(prob.pmr(answers, a, b), b) for b in po if b != a]
            mr = max(v for v, _ in vals)
            cand.append((mr, a, [b for v, b in vals if v > mr - 1e-9]))
        mmr = min(c[0] for c in cand)
        pairs = [(a, b) for mr, a, bs in cand if mr < mmr + 1e-9 for b in bs]
        if tb == "first":
            a, b = pairs[0]
        elif tb in ("far", "near"):
            key = lambda p: np.abs(prob.Y[p[0]] - prob.Y[p[1]]).sum()
            a, b = (max if tb == "far" else min)(pairs, key=key)
        else:
            a, b = pairs[int(rng.integers(len(pairs)))]
        return tuple(int(v) for v in prob.Y[a] - prob.Y[b])
    return strat

rows = []
for line in open(ROOT / "data/alg7_tasks.jsonl"):
    t = json.loads(line)
    if t["group"] != "2":
        continue
    D = [tuple(tuple(l) for l in K) for K in t["D"]]
    cl, _, _ = build_classes(t["n"], D, t["signs"])
    prob = Problem(cl)
    rec = dict(seed=t["seed"], Mprime=prob.Mprime)
    for tb in ("first", "far", "near"):
        rec[tb] = worst_case(prob, make_css(tb))
    rr = [worst_case(prob, make_css("rand", np.random.default_rng(s))) for s in range(5)]
    rec["rand_min"], rec["rand_max"] = min(rr), max(rr)
    rows.append(rec); print(rec, flush=True)
with open(ROOT / "results/css_ties.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
