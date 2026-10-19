"""Проверка теоремы 6 и утверждения 10 (алгоритм 8): 500 случайных задач.
seed = 0..499; n ~ U{4..10}, s ~ U{1..3}, m ~ U{1..5}, коэффициенты ~ U{-5..5},
ДНФ: generators.random_dnf (ранг конъюнкции ~ U{1..n//2})."""
import csv, json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from generators import random_mo
from pareto import pareto_bb, pareto_brute, leaves_points

rows = []
with open(ROOT / "data/alg8_tasks.jsonl", "w") as f:
    for seed in range(500):
        rng = np.random.default_rng(seed)
        n = int(rng.integers(4, 11)); s = int(rng.integers(1, 4)); m = int(rng.integers(1, 6))
        D, c = random_mo(rng, n, s, m)
        f.write(json.dumps(dict(seed=seed, n=n, s=s, D=D, c=c.tolist())) + "\n")
        Par = pareto_brute(n, D, c)
        ok = True
        for fix0 in (True, False):
            leaves, nodes, nconf = pareto_bb(n, D, c, fix_step0=fix0)
            ok &= leaves_points(n, leaves) == Par
            if fix0:
                bound = sum(2 ** (len(set(range(n)) - {i for i, _ in K}
                                      - {i for i in range(n) if not ((c[:, i] > 0).any() and (c[:, i] < 0).any())}) + 1) - 1
                            for K in D)
                rows.append(dict(seed=seed, n=n, s=s, m=m, conf=nconf, nodes=nodes, bound=bound,
                                 ratio=nodes / bound, pareto=len(Par), leaves=len(leaves)))
        assert ok, seed
print("all 500 exact")
with open(ROOT / "results/alg8_rows.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
for s in (1, 2, 3):
    R = [r for r in rows if r["s"] == s]
    print("s=%d: tasks %d, mean nodes/bound %.3f, mean conflicting %.2f" % (s, len(R), np.mean([r["ratio"] for r in R]), np.mean([r["conf"] for r in R])))
