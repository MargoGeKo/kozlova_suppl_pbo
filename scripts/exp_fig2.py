"""Рис. 2 и табл. 1: число операций GAEL и LAEL на ЛПС-цепочке и «широкой»
ЛПС (модель подсчёта -- docstring lps.gael). Результат детерминирован.
Выход: results/fig2.csv, точные формулы и наклоны в results/fig2.txt."""
import csv
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from families import chain, wide  # noqa
from lps import OpCounter, gael, lael  # noqa

VARIANTS = ("scan", "index", "counters")
WS = (10, 20, 40, 80, 160, 320)


def count(S, alg, v):
    o = OpCounter()
    (gael(S, v, o) if alg == "GAEL" else lael(S, ("y", 1), v, o))
    return o.total


def exact_poly(xs, ys, deg=3):
    """Интерполяция по deg+1 точкам в рациональных числах и проверка на
    остальных."""
    xs = [Fraction(x) for x in xs]; ys = [Fraction(y) for y in ys]
    A = [[x ** k for k in range(deg + 1)] for x in xs[:deg + 1]]
    b = ys[:deg + 1]
    n = deg + 1
    for i in range(n):  # Гаусс
        p = next(r for r in range(i, n) if A[r][i] != 0)
        A[i], A[p] = A[p], A[i]; b[i], b[p] = b[p], b[i]
        for r in range(n):
            if r != i and A[r][i] != 0:
                f = A[r][i] / A[i][i]
                A[r] = [a - f * c for a, c in zip(A[r], A[i])]; b[r] -= f * b[i]
    coef = [b[i] / A[i][i] for i in range(n)]
    for x, y in zip(xs, ys):
        assert sum(c * x ** k for k, c in enumerate(coef)) == y
    return coef


def fmt(coef):
    t = []
    for k in range(len(coef) - 1, -1, -1):
        c = coef[k]
        if c == 0:
            continue
        s = ("%s" % c)
        t.append("%s%s" % (s, "" if k == 0 else ("*nu" if k == 1 else "*nu^%d" % k)))
    return " + ".join(t)


rows, txt = [], []
for fam in (chain, wide):
    for alg in ("GAEL", "LAEL"):
        data = {v: [] for v in VARIANTS}
        nus = []
        for w in WS:
            S = fam(w)
            nus.append(S.nu)
            for v in VARIANTS:
                c = count(S, alg, v)
                data[v].append(c)
                rows.append(dict(family=fam.__name__, alg=alg, variant=v, w=w, nu=S.nu,
                                 H=None, ops=c))
        for v in VARIANTS:
            coef = exact_poly(nus, data[v])
            sl = np.polyfit(np.log(nus[3:]), np.log(data[v][3:]), 1)[0]
            txt.append("%-6s %-5s %-9s ops = %s ; slope(nu=%d..%d) = %.2f" %
                       (fam.__name__, alg, v, fmt(coef), nus[3], nus[-1], sl))
# отношение к H*q*nu на цепочке
for w in (80, 160, 320, 640):
    S = chain(w)
    o = OpCounter(); _, H, _, _, _ = gael(S, "scan", o)
    q = len(S.nonterminals())
    txt.append("chain w=%d: scan/(H*q*nu) = %.4f" % (w, o.total / (H * q * S.nu)))
with open(ROOT / "results/fig2.csv", "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
open(ROOT / "results/fig2.txt", "w").write("\n".join(txt) + "\n")
print("\n".join(txt))
for fam in ("chain", "wide"):
    for v in VARIANTS:
        pts = " ".join("(%d,%d)" % (r["nu"], r["ops"]) for r in rows
                       if r["family"] == fam and r["alg"] == "GAEL" and r["variant"] == v)
        print("%s %s: coordinates {%s};" % (fam, v, pts))
