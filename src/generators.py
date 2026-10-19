"""Генераторы случайных задач. Все генераторы детерминированы при заданном
зерне (numpy.random.default_rng(seed)); зерно каждой задачи хранится в
файлах data/*.jsonl вместе с самой задачей."""
from __future__ import annotations

import numpy as np

from lps import LPS


# ------------------------------------------------- G1: ярусные ЛПС (табл. 6)
def tiered_kb(rng, n, H, facts=(2, 4), prods=(1, 3), prem=(2, 3),
              p_fact=0.5, p_neg=0.25, goal_prods=(1, 2)):
    """Ацикличная ярусная ЛПС (разд. 6 статьи).

    n терминальных переменных x1..xn; H ярусов, в каждом равновероятно
    facts[0]..facts[1] промежуточных фактов; у факта prods[0]..prods[1]
    продукций, у продукции prem[0]..prem[1] посылок без повторов. Посылка с
    вероятностью p_fact -- факт одного из двух нижележащих ярусов (ярус и
    факт выбираются равновероятно; на первом ярусе фактов ниже нет), иначе
    терминальный литерал x_i (i равновероятно), с вероятностью p_neg --
    с отрицанием. Цель y выводится goal_prods[0]..goal_prods[1] продукциями,
    посылки которых (2..3, но не больше числа фактов) -- факты верхнего
    яруса."""
    terms = ["x%d" % i for i in range(1, n + 1)]
    tiers, P = [], []
    for t in range(H):
        k = int(rng.integers(facts[0], facts[1] + 1))
        names = [("q%d_%d" % (t + 1, j + 1), 1) for j in range(k)]
        for a in names:
            for _ in range(int(rng.integers(prods[0], prods[1] + 1))):
                r = int(rng.integers(prem[0], prem[1] + 1))
                L = []
                while len(L) < r:
                    lower = tiers[max(0, t - 2):t]
                    if lower and rng.random() < p_fact:
                        tier = lower[int(rng.integers(len(lower)))]
                        b = tier[int(rng.integers(len(tier)))]
                    else:
                        v = terms[int(rng.integers(n))]
                        b = (v, 0 if rng.random() < p_neg else 1)
                    if b not in L and (b[0], 1 - b[1]) not in L:
                        L.append(b)
                P.append((tuple(L), a))
        tiers.append(names)
    top = tiers[-1]
    for _ in range(int(rng.integers(goal_prods[0], goal_prods[1] + 1))):
        r = min(len(top), int(rng.integers(2, 4)))
        idx = rng.choice(len(top), size=r, replace=False)
        P.append((tuple(top[i] for i in sorted(idx)), ("y", 1)))
    return LPS(P, terms)


# ---------------------------- G2: общие ЛПС (утверждения 1, 3: циклы и т.п.)
def general_kb(rng, n=(4, 7), facts=(3, 7), prods=(1, 3), prem=(1, 3),
               p_term=0.5, p_neg_term=0.3, p_neg_fact=0.15, p_back=0.15):
    """ЛПС общего вида с невыводимыми литералами, отрицаниями нетерминальных
    литералов и циклами. Факты q1..qk упорядочены; посылка -- терминальный
    литерал (вероятность p_term, отрицание с вероятностью p_neg_term) или
    факт: с вероятностью 1-p_back -- с меньшим номером, иначе любой
    (это порождает циклы); факт берётся с отрицанием с вероятностью
    p_neg_fact (литерал ~q выводится, только если у него есть продукции:
    у случайной половины фактов с отрицаниями добавляется 1 продукция с
    заключением ~q). Цель y: 1..2 продукции из фактов."""
    nn = int(rng.integers(n[0], n[1] + 1))
    terms = ["x%d" % i for i in range(1, nn + 1)]
    k = int(rng.integers(facts[0], facts[1] + 1))
    facts_ = ["q%d" % i for i in range(1, k + 1)]
    P = []

    def premise(j):
        if j == 0 or rng.random() < p_term:
            v = terms[int(rng.integers(nn))]
            return (v, 0 if rng.random() < p_neg_term else 1)
        if rng.random() < p_back:
            f = facts_[int(rng.integers(k))]
        else:
            f = facts_[int(rng.integers(j))]
        return (f, 0 if rng.random() < p_neg_fact else 1)

    concl = [(f, 1) for f in facts_]
    neg_used = set()
    for j, f in enumerate(facts_):
        for _ in range(int(rng.integers(prods[0], prods[1] + 1))):
            r = int(rng.integers(prem[0], prem[1] + 1))
            L = []
            for _ in range(r):
                b = premise(j)
                if b[0] == f:
                    continue
                if b not in L and (b[0], 1 - b[1]) not in L:
                    L.append(b)
                    if b[1] == 0 and b[0] in facts_:
                        neg_used.add(b[0])
            if L:
                P.append((tuple(L), (f, 1)))
    for f in sorted(neg_used):
        if rng.random() < 0.5:
            j = facts_.index(f)
            L = [premise(j) for _ in range(int(rng.integers(1, 3)))]
            L = [b for b in dict.fromkeys(L) if b[0] != f]
            if L:
                P.append((tuple(L), (f, 0)))
    for _ in range(int(rng.integers(1, 3))):
        r = int(rng.integers(1, 3))
        L = list(dict.fromkeys((facts_[int(rng.integers(k))], 1) for _ in range(r)))
        P.append((tuple(L), ("y", 1)))
    return LPS(P, terms)


# -------------------------- G3: случайная ДНФ (каноническая модель, алг. 7, 8)
def random_dnf(rng, n, m, rank=(1, None)):
    """m конъюнкций; ранг каждой равновероятен в [rank0, rank1]
    (по умолчанию [1, n//2]); переменные без повторов, знак литерала
    равновероятен."""
    r1 = rank[1] if rank[1] is not None else max(1, n // 2)
    D = []
    for _ in range(m):
        r = int(rng.integers(rank[0], r1 + 1))
        idx = sorted(rng.choice(n, size=r, replace=False))
        D.append(tuple((int(i), int(rng.integers(0, 2))) for i in idx))
    return D


def product_classes(rng, block_sizes, zfree=(0, 2)):
    """Задача с множеством классов вида произведения (утверждение 8).
    block_sizes -- список M'_t; блок Gamma_t содержит max(2, M'_t)
    переменных и антицепь Y_t из M'_t векторов веса 1 в ориентированных
    координатах (точка «выбирает» одно мероприятие блока). Добавляются
    0..2 переменные с c_i=0. Знаки остальных коэффициентов случайны;
    ДНФ -- по одной конъюнкции на элемент произведения, фиксирующей все
    переменные из P^{+-} (порядок конъюнкций перемешивается)."""
    import itertools
    Gamma, Y = [], []
    pos = 0
    for Mprime_t in block_sizes:
        size = max(2, Mprime_t)
        Gamma.append(list(range(pos, pos + size)))
        Yt = []
        for e in rng.choice(size, size=Mprime_t, replace=False):
            Yt.append(tuple(1 if i == e else 0 for i in range(size)))
        Y.append(Yt)
        pos += size
    nz = int(rng.integers(zfree[0], zfree[1] + 1))
    n = pos + nz
    signs = [int(rng.choice([1, -1])) for _ in range(pos)] + [0] * nz
    D = []
    for combo in itertools.product(*Y):
        conj = []
        for Gamma_t, yt in zip(Gamma, combo):
            for i, yi in zip(Gamma_t, yt):
                xi = yi if signs[i] > 0 else 1 - yi
                conj.append((i, xi))
        D.append(tuple(conj))
    order = rng.permutation(len(D))
    D = [D[i] for i in order]
    # перестановка переменных, чтобы блоки не шли подряд
    perm = rng.permutation(n)
    D = [tuple(sorted((int(perm[i]), s) for i, s in K_)) for K_ in D]
    sg = [0] * n
    for i in range(n):
        sg[int(perm[i])] = signs[i]
    return n, D, sg


def random_mo(rng, n, s, m, coef=5):
    """Многокритериальная задача (разд. 5): ДНФ random_dnf(n, m) и s
    критериев с целыми коэффициентами из [-coef, coef]."""
    D = random_dnf(rng, n, m)
    c = rng.integers(-coef, coef + 1, size=(s, n))  # c[j, i] = c_{ji}
    return D, c


# ------------------------- G6: произведение классов с изменениями (разд. 3)
def product_variants():
    """Задачи G6 (векторы классов y): произведение единичных блоков с
    размерами M'_t из (2,2), (2,3), (2,2,2), (3,3) (а) без одного класса,
    (б) с одним добавленным вектором той же размерности, (в) с добавленным
    вектором и новой координатой, (г) с двумя добавленными векторами (для
    M' <= 6); перебираются все векторы, сохраняющие попарную
    несравнимость. Возвращает список (имя, вектора классов)."""
    import itertools

    def antichain(Y):
        return not any(all(a >= b for a, b in zip(u, v)) or all(a <= b for a, b in zip(u, v))
                       for u, v in itertools.combinations(Y, 2))

    def product(sizes, pad=0):
        blocks = [[tuple(1 if i == e else 0 for i in range(max(2, s))) for e in range(s)]
                  for s in sizes]
        return [sum(c, ()) + (0,) * pad for c in itertools.product(*blocks)]

    out = []
    for sizes in [(2, 2), (2, 3), (2, 2, 2), (3, 3)]:
        tag = ''.join(map(str, sizes))
        base = product(sizes); d = len(base[0])
        for j in range(len(base)):
            out.append(('minus_%s' % tag, [b for k, b in enumerate(base) if k != j]))
        for v in itertools.product((0, 1), repeat=d):
            if v not in base and antichain(base + [v]):
                out.append(('plus1_%s' % tag, base + [v]))
        base1 = product(sizes, pad=1)
        for v in itertools.product((0, 1), repeat=d):
            if antichain(base1 + [v + (1,)]):
                out.append(('plus1new_%s' % tag, base1 + [v + (1,)]))
        if len(base) <= 6:
            ext = [v for v in itertools.product((0, 1), repeat=d)
                   if v not in base and antichain(base + [v])]
            for u, v in itertools.combinations(ext, 2):
                if antichain(base + [u, v]):
                    out.append(('plus2_%s' % tag, base + [u, v]))
    return [(name, sorted(set(Y))) for name, Y in out]
