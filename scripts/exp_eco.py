"""Модельная задача разд. 6 (выбор природоохранных мероприятий) и
множество Парето на 60 ЛПС табл. 6 с n=12."""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from lps import LPS, ds_algorithm, odv_bitsets, dnf_bitset
from pareto import pareto_bb, pareto_brute, leaves_points
from dnfops import glue_absorb

X = lambda i, s=1: ("x%d" % i, s)
Q = lambda q: (q, 1)
rules = [((X(1),), Q("qs")), ((X(3),), Q("qs")), ((Q("qs"), X(2)), Q("qw")), ((X(1), X(11)), Q("qw")),
         ((X(4), X(6)), Q("qa")), ((X(5), X(7)), Q("qa")), ((X(4), X(5)), Q("qa")),
         ((X(8), X(9)), Q("qo")), ((X(9), X(10)), Q("qo")), ((X(8), X(10)), Q("qo")),
         ((X(11), X(12)), Q("qk")), ((X(3), X(12)), Q("qk")), ((X(3, 0),), Q("qe")), ((X(3), X(10)), Q("qe")),
         ((Q("qw"), Q("qa"), Q("qo"), Q("qk"), Q("qe")), ("y", 1))]
S = LPS(rules, ["x%d" % i for i in range(1, 13)])
e = [3, 5, 4, 4, 3, 2, 1, 3, 2, 3, 1, 1]
k = [40, 120, 90, 60, 35, 10, 15, 50, 8, 45, 12, 10]
dnf, D_size = ds_algorithm(S, ("y", 1))
om = odv_bitsets(S)[("y", 1)]
assert (dnf_bitset(dnf, S.terminals) == om).all()
print("DNF of y:", len(dnf), "conj, lengths", sorted({len(K) for K in dnf}), "|Omega| =", int(om.sum()))
print("intermediate DNF sizes:", {a[0]: v for a, v in D_size.items()})
pos = {v: i for i, v in enumerate(S.terminals)}
D = [tuple(sorted((pos[v], s) for v, s in K)) for K in dnf]
costs = [sum(k[i] for i, s in K if s) for K in D]
j = int(np.argmin(costs)); print("min cost", costs[j], "conj", sorted(D[j]))
# правка комплекса {x2,x4,x6,x9,x12}
xp = {1: 0, 2: 1, 3: 0, 4: 1, 5: 0, 6: 1, 7: 0, 8: 0, 9: 1, 10: 0, 11: 0, 12: 1}
viol = [sum(xp[i + 1] != s for i, s in K) for K in D]
print("cost of x':", sum(k[i - 1] for i in xp if xp[i]), " min edits:", min(viol))
fixes = []
for K, v in zip(D, viol):
    add = [i + 1 for i, s in K if xp[i + 1] != s]
    fixes.append((v, sum(k[i - 1] * (1 if s else -1) for i, s in [(i + 1, s) for i, s in K if xp[i + 1] != s]), add))
print("min-edit fixes:", sorted(f for f in fixes if f[0] == min(viol))[:3])
print("cheapest fix:", min(fixes, key=lambda f: f[1]))
c = np.array([e, [-v for v in k]])
leaves, nodes, nconf = pareto_bb(12, D, c)
Par = pareto_brute(12, D, c)
assert leaves_points(12, leaves) == Par
vals = sorted({(int(np.dot(e, x)), int(np.dot(k, x))) for x in Par})
core = [i + 1 for i in range(12) if len({x[i] for x in Par}) == 1]
red = glue_absorb([frozenset(fx.items()) for fx, _ in leaves])
print("Pareto: nodes", nodes, "leaves", len(leaves), "efficient", len(Par), "E,C range", vals[0], vals[-1],
      "after gluing", len(red), "core", {i: next(iter(Par))[i - 1] for i in core})
print("E>=24 needed for x2/x3:", min(int(np.dot(e, x)) for x in Par if x[1] or x[2]))

# 60 ЛПС табл. 6 с n=12
rows = []
for line in open(ROOT / "data/table6_kb.jsonl"):
    t = json.loads(line)
    if t["n"] != 12:
        continue
    lit = lambda s: (s[1:], 0) if s.startswith("~") else (s, 1)
    Sk = LPS([(tuple(lit(b) for b in L), lit(a)) for L, a in t["prods"]], t["terminals"])
    dnf, _ = ds_algorithm(Sk, ("y", 1))
    pos = {v: i for i, v in enumerate(Sk.terminals)}
    D = [tuple(sorted((pos[v], s) for v, s in K)) for K in dnf]
    rng = np.random.default_rng(t["seed"] + 1)
    ee = rng.integers(1, 6, size=12); kk = rng.integers(5, 101, size=12)
    c = np.array([ee, -kk])
    leaves, nodes, _ = pareto_bb(12, D, c)
    Par = pareto_brute(12, D, c)
    assert leaves_points(12, leaves) == Par
    red = glue_absorb([frozenset(fx.items()) for fx, _ in leaves])
    core = sum(1 for i in range(12) if len({x[i] for x in Par}) == 1)
    om = int(dnf_bitset(dnf, Sk.terminals).sum())
    rows.append((t["H"], om, len(Par), len(red), core))
for H in (2, 3, 4):
    R = np.array([r[1:] for r in rows if r[0] == H])
    print("H=%d: median |Omega| %g, efficient %g, DNF after gluing %g, core vars %g" % ((H,) + tuple(np.median(R, 0))))
R = np.array([r[1:] for r in rows])
print("all 60: median", np.median(R, 0))
