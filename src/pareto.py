r"""Алгоритм 8: множество Парето задачи max(f_1..f_s) при D=1 методом
ветвей и границ, с подсчётом вершин.

Имена: c[j, i] = c_{ji}; Pt0_plus, Pt0_minus -- \widetilde P_0^{+}, \widetilde P_0^{-};
Z0 -- Z_0; conf -- Conf; f_lo, f_hi -- векторные оценки \underline f(X), \overline f(X)."""
from __future__ import annotations

import itertools

import numpy as np


def dominates(u, v):
    return all(a >= b for a, b in zip(u, v)) and tuple(u) != tuple(v)


class Records:
    def __init__(self):
        self.R = []

    def add(self, v):
        v = tuple(int(a) for a in v)
        if any(dominates(r, v) or r == v for r in self.R):
            return
        self.R = [r for r in self.R if not dominates(v, r)]
        self.R.append(v)

    def dominated(self, v):
        v = tuple(v)
        return any(dominates(r, v) for r in self.R)


def pareto_bb(n, D, c, fix_step0=True, branch="first"):
    """D: список конъюнкций [(i, sigma)], c: массив s x n коэффициентов c_{ji}.
    Возвращает (листья [(dict фиксаций, вектор)], число вершин)."""
    c = np.asarray(c)
    s = c.shape[0]
    Pt0_plus = {i for i in range(n) if (c[:, i] >= 0).all() and (c[:, i] > 0).any()}
    Pt0_minus = {i for i in range(n) if (c[:, i] <= 0).all() and (c[:, i] < 0).any()}
    Z0 = {i for i in range(n) if (c[:, i] == 0).all()}
    conf = [i for i in range(n) if i not in Pt0_plus | Pt0_minus | Z0]

    def f_bounds(fix):
        free = [i for i in range(n) if i not in fix]
        base = np.array([sum(c[j, i] * fix[i] for i in fix) for j in range(s)])
        f_hi = base + np.array([sum(max(c[j, i], 0) for i in free) for j in range(s)])
        f_lo = base + np.array([sum(min(c[j, i], 0) for i in free) for j in range(s)])
        pts = []
        for j in range(s):
            x = dict(fix)
            for i in free:
                x[i] = 1 if c[j, i] > 0 else 0
            pts.append(tuple(int(c[k] @ np.array([x[i] for i in range(n)])) for k in range(s)))
        return f_lo, f_hi, free, pts

    R = Records()
    A = []
    for K in D:
        fix = dict(K)
        if any(fix.get(i, s_) != s_ for i, s_ in K):
            continue
        if fix_step0:
            for i in Pt0_plus:
                fix.setdefault(i, 1)
            for i in Pt0_minus:
                fix.setdefault(i, 0)
        f_lo, f_hi, free, pts = f_bounds(fix)
        for p in pts:
            R.add(p)
        A.append((fix, f_lo, f_hi, free))
    nodes = len(A)
    leaves = []
    while A:
        fix, f_lo, f_hi, free = A.pop()
        if R.dominated(f_hi):
            continue
        cf = [i for i in free if i not in Z0]
        if not cf:
            leaves.append((fix, tuple(int(a) for a in f_hi)))
            continue
        i = cf[0]
        for b in (0, 1):
            fix2 = dict(fix); fix2[i] = b
            f_lo2, f_hi2, free2, pts = f_bounds(fix2)
            for p in pts:
                R.add(p)
            A.append((fix2, f_lo2, f_hi2, free2))
            nodes += 1
    out = [(fx, v) for fx, v in leaves if not R.dominated(v)]
    return out, nodes, len(conf)


def pareto_brute(n, D, c):
    c = np.asarray(c)
    X = np.array(list(itertools.product((0, 1), repeat=n)))
    ok = np.zeros(len(X), dtype=bool)
    for K in D:
        t = np.ones(len(X), dtype=bool)
        for i, s_ in K:
            t &= X[:, i] == s_
        ok |= t
    X = X[ok]
    V = X @ c.T
    eff = []
    for k in range(len(X)):
        if not any((V[j] >= V[k]).all() and (V[j] != V[k]).any() for j in range(len(X))):
            eff.append(tuple(X[k]))
    return set(eff)


def leaves_points(n, leaves):
    pts = set()
    for fix, _ in leaves:
        free = [i for i in range(n) if i not in fix]
        for bits in itertools.product((0, 1), repeat=len(free)):
            x = dict(fix); x.update(zip(free, bits))
            pts.add(tuple(x[i] for i in range(n)))
    return pts
