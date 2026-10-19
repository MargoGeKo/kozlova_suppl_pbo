r"""Логические продукционные системы (ЛПС): представление, вывод, ОДВ,
алгоритмы GAEL/LAEL (с подсчётом операций), D- и DS-алгоритмы.

Литерал -- пара (имя переменной, знак), знак 1 -- x, 0 -- \bar x.
Продукция -- пара (кортеж посылок, заключение).
Терминальные переменные перечислены в lps.terminals (их порядок задаёт
порядок разрядов наборов при переборе).

Модель подсчёта операций (рис. 2 статьи) описана в docstring функции gael().

Имена следуют обозначениям статьи: nu -- число продукций nu(S),
Lam -- множество Lambda строки D-алгоритма, D_size[x] -- |D_x|.
"""
from __future__ import annotations

import itertools
from collections import defaultdict, deque

import numpy as np


def neg(lit):
    return (lit[0], 1 - lit[1])


def lit_str(lit):
    return lit[0] if lit[1] else "~" + lit[0]


class LPS:
    def __init__(self, prods, terminals):
        # порядок продукций и порядок посылок внутри продукции сохраняются:
        # от них зависит (детерминированный) подсчёт операций
        self.prods = [(tuple(L), a) for L, a in prods]
        self.terminals = list(terminals)
        self.tset = set(self.terminals)

    def is_term(self, lit):
        return lit[0] in self.tset

    @property
    def nu(self):  # nu(S) -- число продукций
        return len(self.prods)

    @property
    def size(self):  # |S| = sum(|L_j|+1)
        return sum(len(L) + 1 for L, _ in self.prods)

    def nonterminals(self):
        s = set()
        for L, a in self.prods:
            s.add(a)
            s.update(b for b in L if not self.is_term(b))
        return s

    def by_concl(self):
        idx = defaultdict(list)
        for j, (L, a) in enumerate(self.prods):
            idx[a].append(j)
        return idx

    def sub(self, prod_ids):
        return LPS([self.prods[j] for j in sorted(prod_ids)], self.terminals)


# ---------------------------------------------------------------- вывод / ОДВ
def odv_bitsets(S: LPS, terms=None):
    """Области выводимости всех литералов как булевы массивы длины 2^n
    (разряд i набора -- значение terms[i]). Вычисляется наименьшая
    неподвижная точка прямого вывода, параллельно по всем наборам."""
    terms = list(terms if terms is not None else S.terminals)
    n = len(terms)
    idx = np.arange(1 << n, dtype=np.int64)
    val = {}
    for i, v in enumerate(terms):
        b = ((idx >> i) & 1).astype(bool)
        val[(v, 1)] = b
        val[(v, 0)] = ~b
    zero = np.zeros(1 << n, dtype=bool)
    for a in S.nonterminals():
        val[a] = zero.copy()
    changed = True
    while changed:
        changed = False
        for L, a in S.prods:
            t = np.ones(1 << n, dtype=bool)
            for b in L:
                t &= val[b]
            new = val[a] | t
            if (new != val[a]).any():
                val[a] = new
                changed = True
    return val


def dnf_bitset(dnf, terms):
    n = len(terms)
    pos = {v: i for i, v in enumerate(terms)}
    idx = np.arange(1 << n, dtype=np.int64)
    out = np.zeros(1 << n, dtype=bool)
    for K in dnf:
        t = np.ones(1 << n, dtype=bool)
        for v, s in K:
            bit = ((idx >> pos[v]) & 1).astype(bool)
            t &= bit if s else ~bit
        out |= t
    return out


def consistent(S: LPS):
    """Непротиворечивость: ни на одном наборе не выводятся b и ~b."""
    val = odv_bitsets(S)
    for a in S.nonterminals():
        na = neg(a)
        if na in val and (val[a] & val[na]).any():
            return False
    return True


# ------------------------------------------------------------- GAEL / LAEL
class OpCounter:
    def __init__(self):
        self.view = 0   # просмотр записи продукции при поиске
        self.check = 0  # проверка одной посылки (или уменьшение счётчика)

    @property
    def total(self):
        return self.view + self.check


def step0_counters(S: LPS, ops: OpCounter | None = None):
    """Шаг 0 GAEL (удаление невыводимого) со счётчиками невыполненных
    посылок (схема Даулинга--Галье), O(|S|)."""
    cnt = []
    occ = defaultdict(list)  # нетерминальный литерал -> номера продукций, где он посылка
    q = deque()
    V = set()
    for j, (L, a) in enumerate(S.prods):
        if ops: ops.view += 1
        nt = [b for b in L if not S.is_term(b)]
        cnt.append(len(nt))
        for b in nt:
            occ[b].append(j)
        if not nt and a not in V:
            V.add(a); q.append(a)
    while q:
        b = q.popleft()
        for j in occ[b]:
            if ops: ops.check += 1
            cnt[j] -= 1
            if cnt[j] == 0:
                a = S.prods[j][1]
                if a not in V:
                    V.add(a); q.append(a)
    allnt = S.nonterminals()
    dead = allnt - V
    keep = [j for j, (L, a) in enumerate(S.prods)
            if all(S.is_term(b) or b in V for b in L)]
    return V, dead, S.sub(keep)


def gael(S: LPS, variant="counters", ops: OpCounter | None = None):
    """GAEL (алгоритм 2).  variant: 'scan' | 'index' | 'counters'.

    Модель подсчёта операций (рис. 2):
      * шаг 0 во всех вариантах выполняется со счётчиками: 1 операция на
        каждую продукцию (инициализация) и 1 на каждое уменьшение счётчика;
      * 'scan':  на каждом проходе i для каждого литерала a из W
        просматриваются все nu(S') записей S' (каждая -- 1 операция 'view');
        для каждой найденной продукции L->a проверяются посылки в порядке
        записи до первой невыполненной (каждая проверка -- 1 операция
        'check'); проверка литерала a прекращается на первой продукции
        с невыполненной посылкой;
      * 'index': то же, но просматриваются только записи продукций с
        заключением a (индекс по заключению);
      * 'counters': для каждого литерала хранится число нетерминальных
        вхождений в посылки его продукций, ещё не отнесённых к слоям;
        1 операция на инициализацию каждой продукции и 1 на каждое
        уменьшение счётчика.
    Возвращает (слои Re_0..Re_H, H, Dead, Cyc)."""
    V, dead, Sp = step0_counters(S, ops)
    idx = Sp.by_concl()
    level = {}  # нетерминальный литерал -> номер слоя снизу (высота)
    if variant in ("scan", "index"):
        W = [a for a in _ordered_nonterminals(Sp) if a in V]
        i = 0
        while True:
            i += 1
            new = []
            for a in W:
                if variant == "scan":
                    if ops: ops.view += Sp.nu
                    my = idx[a]
                else:
                    if ops: ops.view += len(idx[a])
                    my = idx[a]
                ok = True
                for j in my:
                    for b in Sp.prods[j][0]:
                        if ops: ops.check += 1
                        if not (Sp.is_term(b) or level.get(b, i) < i):
                            ok = False
                            break
                    if not ok:
                        break
                if ok:
                    new.append(a)
            for a in new:
                level[a] = i
            W = [a for a in W if a not in level]
            if not new:
                break
        H = i - 1
        cyc = set(W)
    elif variant == "counters":
        need = defaultdict(int)
        occ = defaultdict(list)
        for j, (L, a) in enumerate(Sp.prods):
            if ops: ops.view += 1
            for b in L:
                if not Sp.is_term(b):
                    need[a] += 1
                    occ[b].append(a)
        cur = [a for a in _ordered_nonterminals(Sp) if a in V and need[a] == 0]
        i = 0
        while cur:
            i += 1
            nxt = []
            for a in cur:
                level[a] = i
            for a in cur:
                for c in occ[a]:
                    if ops: ops.check += 1
                    need[c] -= 1
                    if need[c] == 0:
                        nxt.append(c)
            cur = nxt
        H = i
        cyc = {a for a in V if a not in level}
    else:
        raise ValueError(variant)
    layers = [set() for _ in range(H + 1)]
    for a, h in level.items():
        layers[H - h].add(a)
    # Re_H дополнительно содержит терминальные литералы S'
    for L, a in Sp.prods:
        for b in L:
            if Sp.is_term(b):
                layers[H].add(b)
    return layers, H, dead, cyc, Sp


def _ordered_nonterminals(S):
    """Детерминированный порядок нетерминальных литералов: по первому
    появлению в записи ЛПС (заключения раньше посылок той же продукции)."""
    seen, out = set(), []
    for L, a in S.prods:
        for b in (a,) + L:
            if not S.is_term(b) and b not in seen:
                seen.add(b); out.append(b)
    return out


def heights(S: LPS):
    """Высоты по определению (рекурсия), None -- не определена."""
    idx = S.by_concl()
    memo, state = {}, {}

    def h(a):
        if S.is_term(a):
            return 0
        if a in memo:
            return memo[a]
        if state.get(a) == 1:
            return None
        state[a] = 1
        ch = [b for j in idx[a] for b in S.prods[j][0]]
        if not idx[a]:
            r = None
        else:
            hs = [h(b) for b in ch] if ch else [0]
            r = None if any(x is None for x in hs) else 1 + max(hs)
        state[a] = 2
        memo[a] = r
        return r
    return {a: h(a) for a in S.nonterminals()}


def derive_subsystem(S: LPS, z, variant="index", ops: OpCounter | None = None):
    """Выводящая подсистема (шаги 1--2 LAEL): спуск от z по продукциям."""
    idx = S.by_concl()
    W, O, keep = {z}, deque([z]), set()
    while O:
        a = O.popleft()
        if variant == "scan":
            if ops: ops.view += S.nu
        else:
            if ops: ops.view += len(idx[a])
        for j in idx[a]:
            keep.add(j)
            for b in S.prods[j][0]:
                if ops: ops.check += 1
                if not S.is_term(b) and b not in W:
                    W.add(b); O.append(b)
    return S.sub(keep)


def lael(S: LPS, z, variant="counters", ops: OpCounter | None = None):
    """LAEL (алгоритм 3). Возвращает (Sz, слои, H, статус)."""
    v1 = "scan" if variant == "scan" else "index"
    Sz = derive_subsystem(S, z, v1, ops)
    V, dead, Sp = step0_counters(Sz, ops)
    if z in dead or z not in V:
        return Sz, [], 0, "dead"
    Sz = derive_subsystem(Sp, z, "index", ops)
    layers, H, dead2, cyc, Sz2 = gael(Sz, variant, ops)
    if z in cyc:
        return Sz2, layers, H, "cycle"
    return Sz2, layers, H, "ok"


# ------------------------------------------------------------- D-алгоритм
class DStats:
    def __init__(self):
        self.written = 0    # строк, записанных в «Открыт»
        self.completed = 0  # завершённых строк (Lambda пусто, шаг 6)


def d_algorithm(S: LPS, z, stats: DStats | None = None):
    """Уточнённый D-алгоритм (алгоритм 1). Возвращает список конъюнкций
    (frozenset терминальных литералов). Строка -- пара (T, Lam),
    Lam -- список пар (литерал, Pr(литерал))."""
    idx = S.by_concl()
    open_, closed = [], []

    def write(T, Lam):
        tv = {}
        for v, s in T:
            if tv.get(v, s) != s:
                return
            tv[v] = s
        lits = set(T) | {b for b, _ in Lam}
        for b, _ in Lam:
            if neg(b) in lits:
                return
        for b, _ in Lam:
            if not idx.get(b):
                return
        if stats: stats.written += 1
        open_.append((frozenset(T), tuple(Lam)))

    for j in idx.get(z, []):
        L = S.prods[j][0]
        T = [b for b in L if S.is_term(b)]
        Lam = []
        for b in L:
            if not S.is_term(b) and all(b != c for c, _ in Lam):
                Lam.append((b, frozenset({z})))
        write(T, Lam)
    while open_:
        T, Lam = open_.pop()
        if not Lam:
            if stats: stats.completed += 1
            if not any(K <= T for K in closed):
                closed = [K for K in closed if not K >= T]
                closed.append(T)
            continue
        a, Pr_a = Lam[0]
        A = Pr_a | {a}
        Lam_rest = list(Lam[1:])
        for j in idx.get(a, []):
            L = S.prods[j][0]
            if any(b in A for b in L):
                continue
            T2 = set(T) | {b for b in L if S.is_term(b)}
            Lam2 = list(Lam_rest)
            for b in L:
                if S.is_term(b):
                    continue
                for k, (c, Pr_c) in enumerate(Lam2):
                    if c == b:
                        Lam2[k] = (c, Pr_c | A)
                        break
                else:
                    Lam2.append((b, A))
            write(T2, Lam2)
    return closed


def ds_algorithm(S: LPS, z, stats: DStats | None = None, trace=None):
    """Уточнённый DS-алгоритм (алгоритм 4). trace(Scur, x) вызывается
    после каждой замены шага 4.1в (для проверки утверждения 3).
    Возвращает (ДНФ ОДВ(z), D_size), D_size[x] = |D_x|."""
    Sz, layers, H, status = lael(S, z)
    if status == "dead":
        return [], {}
    if status == "cycle":
        return d_algorithm(Sz, z, stats), {}
    cur = Sz
    D_size = {}  # |D_x| для обработанных литералов x
    for k in range(H - 1, -1, -1):
        for x in sorted(layers[k] - set(l for l in layers[k] if cur.is_term(l)),
                        key=lit_str):
            Sx = derive_subsystem(cur, x)
            dnf = d_algorithm(Sx, x, stats)
            D_size[x] = len(dnf)
            rest = [p for p in cur.prods if p[1] != x]
            cur = LPS(rest + [(tuple(sorted(K)), x) for K in dnf], cur.terminals)
            if trace:
                trace(cur, x)
        cur = derive_subsystem(cur, z)
    return [frozenset(L) for L, a in cur.prods if a == z], D_size
