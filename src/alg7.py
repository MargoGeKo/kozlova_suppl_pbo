"""Алгоритм 7: сужающие запросы при известных знаках коэффициентов.

Состояние знаний -- кортеж ответов ((d, rho), ...), d -- вектор
разности ориентированных координат на P^{+-} = P^+ u P^-, rho in {'>','<','='}.
Имена: Ppm -- P^{+-}, Mprime -- M' (число классов), dim -- |P^{+-}|,
supp -- |supp d|, prob -- экземпляр задачи (класс Problem).
Все проверки (а), (б) и PMR -- задачи ЛП, решаемые HiGHS через
scipy.optimize.linprog(method='highs')."""
from __future__ import annotations

import itertools
from functools import lru_cache

import numpy as np
from scipy.optimize import linprog

LP_CALLS = [0]


def build_PiA(n, D, sg):
    """Экстремальные точки интервалов (с учётом всех значений Delta),
    затем удаление мажорируемых. Возвращает список различных векторов
    ориентированных координат классов (M' штук) и точки Pi(A)."""
    A = set()
    for K in D:
        fix = dict(K)
        free = [i for i in range(n) if i not in fix]
        zero = [i for i in free if sg[i] == 0]
        for bits in itertools.product((0, 1), repeat=len(zero)):
            x = dict(fix)
            for i in free:
                if sg[i] > 0: x[i] = 1
                elif sg[i] < 0: x[i] = 0
            x.update(zip(zero, bits))
            A.add(tuple(x[i] for i in range(n)))
    Ppm = [i for i in range(n) if sg[i] != 0]  # P^{+-}

    def y(x):
        return tuple(x[i] if sg[i] > 0 else 1 - x[i] for i in Ppm)
    Y = {x: y(x) for x in A}
    Pi = [x for x in A if not any(all(a >= b for a, b in zip(Y[z], Y[x])) and Y[z] != Y[x] for z in A)]
    classes = sorted(set(Y[x] for x in Pi))
    return classes, Pi, Ppm


def build_classes(n, D, sg):
    """Классы Pi(A) без перебора значений Delta (замечание 3 статьи).

    Все экстремальные точки интервала Omega_j совпадают на P^{+-}: там они
    равны либо литералам K_j, либо значению, заданному знаком c_i; значения
    Delta меняют лишь координаты из P^0. Поэтому каждый интервал даёт один
    вектор y_j, мажорирование зависит только от векторов y, и
        Pi(A) = объединение интервалов K_j^* по недоминируемым y_j,
    где K_j^* -- конъюнкция K_j, дополненная литералами свободных
    переменных из P^{+-} (x_i = 1 при c_i > 0, x_i = 0 при c_i < 0).
    Число классов M' <= m, время O(m^2 n).
    Возвращает (classes, Ppm, dnf_Pi): classes -- отсортированные
    недоминируемые векторы y, dnf_Pi[y] -- список конъюнкций K_j^* класса y."""
    Ppm = [i for i in range(n) if sg[i] != 0]
    ys, conj = [], []
    for K in D:
        fix = dict(K)
        if any(fix.get(i, s_) != s_ for i, s_ in K):
            continue
        star = dict(fix)
        for i in Ppm:
            if i not in star:
                star[i] = 1 if sg[i] > 0 else 0
        ys.append(tuple(star[i] if sg[i] > 0 else 1 - star[i] for i in Ppm))
        conj.append(tuple(sorted(star.items())))
    nd = [j for j, yj in enumerate(ys)
          if not any(all(a >= b for a, b in zip(yk, yj)) and yk != yj for yk in ys)]
    dnf_Pi = {}
    for j in nd:
        dnf_Pi.setdefault(ys[j], [])
        if conj[j] not in dnf_Pi[ys[j]]:
            dnf_Pi[ys[j]].append(conj[j])
    return sorted(dnf_Pi), Ppm, dnf_Pi


def _lp_feasible(A_ub, b_ub, A_eq, b_eq, dim):
    LP_CALLS[0] += 1
    res = linprog(np.zeros(dim), A_ub=A_ub if len(A_ub) else None, b_ub=b_ub if len(b_ub) else None,
                  A_eq=A_eq if len(A_eq) else None, b_eq=b_eq if len(b_eq) else None,
                  bounds=[(1, None)] * dim, method="highs")
    return res.status == 0


def _system(answers, dim):
    A_ub, b_ub, A_eq, b_eq = [], [], [], []
    for d, r in answers:
        d = np.asarray(d, float)
        if r == ">":
            A_ub.append(-d); b_ub.append(-1)
        elif r == "<":
            A_ub.append(d); b_ub.append(-1)
        else:
            A_eq.append(d); b_eq.append(0)
    return A_ub, b_ub, A_eq, b_eq


class Problem:
    def __init__(self, classes):
        self.Y = [np.array(c, float) for c in classes]
        self.dim = len(classes[0]) if classes else 0  # |P^{+-}|
        self.Mprime = len(classes)  # M'
        self._po = {}

    def consistent(self, answers):
        A_ub, b_ub, A_eq, b_eq = _system(answers, self.dim)
        return _lp_feasible(np.array(A_ub), np.array(b_ub), np.array(A_eq), np.array(b_eq), self.dim)

    def PO(self, answers):
        """Индексы возможно оптимальных классов и разбиение на классы
        равноценности; ключ -- множество ответов (порядок не важен)."""
        key = frozenset(answers)
        if key in self._po:
            return self._po[key]
        base = _system(answers, self.dim)
        po = []
        for a in range(self.Mprime):
            A_ub, b_ub, A_eq, b_eq = (list(t) for t in base)
            for b in range(self.Mprime):
                if b != a:
                    A_ub.append(-(self.Y[a] - self.Y[b])); b_ub.append(0)
            if _lp_feasible(np.array(A_ub), np.array(b_ub), np.array(A_eq), np.array(b_eq), self.dim):
                po.append(a)
        # равноценность: разность в линейной оболочке ответов «=»
        E = [np.asarray(d, float) for d, r in answers if r == "="]
        rE = np.linalg.matrix_rank(np.array(E)) if E else 0
        groups = []
        for a in po:
            for g in groups:
                v = self.Y[a] - self.Y[g[0]]
                if np.allclose(v, 0) or (E and np.linalg.matrix_rank(np.vstack(E + [v])) == rE):
                    g.append(a); break
            else:
                groups.append([a])
        self._po[key] = (po, groups)
        return po, groups

    def kappa(self, answers):
        return len(self.PO(answers)[1])

    def queries(self, answers):
        """Неизбыточные запросы о парах точек разных классов PO
        (векторы d с точностью до знака), в порядке перечисления пар."""
        po, groups = self.PO(answers)
        seen, out = set(), []
        reps = [g[0] for g in groups]
        for i, j in itertools.combinations(range(len(reps)), 2):
            d = tuple(int(v) for v in self.Y[reps[i]] - self.Y[reps[j]])
            key = d if d > tuple(-v for v in d) else tuple(-v for v in d)
            if key in seen:
                continue
            seen.add(key)
            R = [r for r in (">", "<", "=") if self.consistent(answers + ((key, r),))]
            if len(R) >= 2:
                out.append((key, R))
        return out

    # ---------------------------------------------- минимаксное сожаление
    def pmr(self, answers, a, b):
        A_ub, b_ub, A_eq, b_eq = _system(answers, self.dim)
        # замыкание конуса: правые части 0 вместо +-1, w >= 0, sum w = 1
        A_ub = [row for row in A_ub]; b_ub = [0.0] * len(A_ub)
        A_eq = list(A_eq) + [np.ones(self.dim)]; b_eq = list(b_eq) + [1.0]
        LP_CALLS[0] += 1
        res = linprog(-(self.Y[b] - self.Y[a]), A_ub=np.array(A_ub) if A_ub else None,
                      b_ub=np.array(b_ub) if b_ub else None, A_eq=np.array(A_eq), b_eq=np.array(b_eq),
                      bounds=[(0, None)] * self.dim, method="highs")
        return -res.fun

    def mmr(self, answers):
        po, groups = self.PO(answers)
        best = None
        for a in po:
            mr, adv = -1, None
            for b in po:
                if b == a:
                    continue
                v = self.pmr(answers, a, b)
                if v > mr + 1e-9:
                    mr, adv = v, b
            if mr < 0:
                mr = 0.0
            if best is None or mr < best[0] - 1e-9:
                best = (mr, a, adv)
        return best


# ----------------------------------------------------------------- стратегии
def strategy_g(prob, answers):
    """Правило (6): максимум g_k(d), при равенстве -- минимум |supp d|."""
    kap = prob.kappa(answers)
    best = None
    for d, R in prob.queries(answers):
        g = kap - max(prob.kappa(answers + ((d, r),)) for r in R)
        supp = sum(1 for v in d if v)
        key = (-g, supp)
        if best is None or key < best[0]:
            best = (key, d)
    return best[1]


def make_sequential(order):
    def strat(prob, answers):
        po, groups = prob.PO(answers)
        rank = {a: order.index(a) for a in po}
        gr = sorted(groups, key=lambda g: min(rank[a] for a in g))
        lead, chal = gr[0][0], gr[1][0]
        return tuple(int(v) for v in prob.Y[lead] - prob.Y[chal])
    return strat


def strategy_css(prob, answers):
    mr, a, b = prob.mmr(answers)
    return tuple(int(v) for v in prob.Y[a] - prob.Y[b])


def worst_case(prob, strat, answers=(), depth=0, limit=20):
    """Наибольшее число запросов, когда ответы выбирает противник из
    совместных."""
    if prob.kappa(answers) == 1:
        return 0
    d = strat(prob, answers)
    R = [r for r in (">", "<", "=") if prob.consistent(answers + ((d, r),))]
    assert len(R) >= 2, "избыточный запрос"
    return 1 + max(worst_case(prob, strat, answers + ((d, r),), depth + 1) for r in R)


def optimal_at_most(prob, answers, depth, memo):
    """Можно ли гарантированно закончить не более чем за depth запросов
    (уравнение Беллмана, запросы -- попарные сравнения)."""
    kap = prob.kappa(answers)
    if kap == 1:
        return True
    if depth == 0 or kap > 2 ** depth:
        return False
    key = (frozenset(answers), depth)
    if key in memo:
        return memo[key]
    ok = False
    for d, R in prob.queries(answers):
        if all(optimal_at_most(prob, answers + ((d, r),), depth - 1, memo) for r in R):
            ok = True; break
    memo[key] = ok
    return ok


def optimal(prob):
    memo = {}
    lo = max(0, int(np.ceil(np.log2(prob.Mprime))))
    for depth in range(lo, prob.Mprime):
        if optimal_at_most(prob, (), depth, memo):
            return depth
    return prob.Mprime - 1


# ------------------------------------------------- противоречивые ответы
def conflict_set(prob, answers):
    """Минимальное противоречивое подмножество ответов (IIS) при условиях
    знаков w_i >= 1, найденное фильтром удаления: ответ исключается, если
    без него система остаётся несовместной. Не более len(answers) задач ЛП.
    Возвращает индексы ответов, входящих в конфликт."""
    assert not prob.consistent(tuple(answers)), "ответы совместны"
    keep = list(range(len(answers)))
    for i in range(len(answers)):
        trial = [j for j in keep if j != i]
        if not prob.consistent(tuple(answers[j] for j in trial)):
            keep = trial
    return keep


def max_consistent(prob, answers):
    """Наибольшее совместное подмножество ответов (перебор по числу
    удаляемых ответов; число ответов не превосходит M'-1)."""
    k = len(answers)
    for drop in range(k + 1):
        for rem in itertools.combinations(range(k), drop):
            sub = tuple(a for j, a in enumerate(answers) if j not in rem)
            if prob.consistent(sub):
                return [j for j in range(k) if j not in rem]
    return []


def worst_case_ties(prob, answers=(), memo=None):
    """Худший случай правила (6) при наихудшем разрешении ничьих, которые
    остаются после критериев g_k(d) и |supp d|."""
    if memo is None:
        memo = {}
    key = frozenset(answers)
    if key in memo:
        return memo[key]
    kap = prob.kappa(answers)
    if kap == 1:
        memo[key] = 0
        return 0
    best, ties = None, []
    for d, R in prob.queries(answers):
        g = kap - max(prob.kappa(answers + ((d, r),)) for r in R)
        k2 = (-g, sum(1 for v in d if v))
        if best is None or k2 < best:
            best, ties = k2, [(d, R)]
        elif k2 == best:
            ties.append((d, R))
    v = max(1 + max(worst_case_ties(prob, answers + ((d, r),), memo) for r in R) for d, R in ties)
    memo[key] = v
    return v
