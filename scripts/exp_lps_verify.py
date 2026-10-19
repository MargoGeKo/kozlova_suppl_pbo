"""Вычислительная проверка утверждений 1, 3 (GAEL/LAEL/DS) и 4 (оценки DS).

Часть A (утв. 1, 3): зёрна 0, 1, ... генератора general_kb; берутся
непротиворечивые ЛПС с выводимой целью y, пока не набрано N_A = 3000.
Для каждой: (1) Dead, слои и Cyc алгоритма 2 во всех трёх вариантах совпадают
с найденными по определению; (2) LAEL для z=y; (3) при статусе «норма»
после каждой замены шага 4.1в ОДВ всех литералов текущей ЛПС совпадают с
исходными; (4) итоговая ДНФ = ОДВ(y), найденная прямым выводом.

Часть B (утв. 4; kv, ka -- k_vee, k_&): 300 ацикличных ярусных ЛПС (n=8..12, H=2..4, зёрна
50000+k) -- неравенства |D_x| <= G(x) <= tau(x) <= tau(z), G(x) <= mu_h;
семейство tree(kv, ka, H) -- |D_z| = mu_H; семейство dvsds(H) --
2^{H-1} завершённых строк D-алгоритма против 2H-1 у DS."""
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lps import (LPS, DStats, OpCounter, consistent, d_algorithm,  # noqa
                 derive_subsystem, dnf_bitset, ds_algorithm, gael, heights,
                 lael, odv_bitsets, step0_counters)
from generators import general_kb, tiered_kb  # noqa
from families import dvsds, tree  # noqa


def scc_cyc(S):
    """Литералы на циклах графа «посылка -> заключение» и достижимые из них."""
    G = defaultdict(set)
    for L, a in S.prods:
        for b in L:
            if not S.is_term(b):
                G[b].add(a)
    nodes = S.nonterminals()
    on_cycle = set()
    for v in nodes:  # v на цикле <=> v достижима из своих преемников
        st, seen = list(G[v]), set()
        while st:
            u = st.pop()
            if u == v:
                on_cycle.add(v); break
            if u in seen:
                continue
            seen.add(u); st.extend(G[u])
    res, st = set(on_cycle), list(on_cycle)
    while st:
        u = st.pop()
        for w in G[u]:
            if w not in res:
                res.add(w); st.append(w)
    return res


def part_a(NA=3000):
    stats = defaultdict(int)
    hmax = 0
    seed = 0
    while stats["inst"] < NA:
        rng = np.random.default_rng(seed); seed += 1
        S = general_kb(rng)
        val0 = odv_bitsets(S)
        if ("y", 1) not in val0 or not val0[("y", 1)].any() or not consistent(S):
            continue
        stats["inst"] += 1
        V, dead, Sp = step0_counters(S)
        dead_def = {a for a in S.nonterminals() if a not in V}
        assert dead == dead_def
        stats["with_dead"] += bool(dead)
        stats["with_neg_fact"] += any(a[1] == 0 for a in S.nonterminals())
        hdef = heights(Sp)
        cyc_def = scc_cyc(Sp) & V
        for v in ("scan", "index", "counters"):
            layers, H, d2, cyc, _ = gael(S, v)
            assert cyc == cyc_def, (seed, v)
            hs = {a: h for a, h in hdef.items() if h is not None}
            assert H == (max(hs.values()) if hs else 0)
            for k, Lk in enumerate(layers):
                assert {a for a in Lk if not Sp.is_term(a)} == \
                    {a for a, h in hs.items() if h == H - k}
        stats["with_cycles"] += bool(cyc_def)
        Sz, layers, Hz, status = lael(S, ("y", 1))
        if status == "cycle":
            stats["goal_on_cycle"] += 1
            assert ("y", 1) in scc_cyc(Sz)
        else:
            assert status == "ok"
            hmax = max(hmax, Hz)
            assert layers[0] == {("y", 1)}

            def trace(cur, x):
                v1 = odv_bitsets(cur)
                for a in cur.nonterminals():
                    assert (v1[a] == val0[a]).all(), (seed, x, a)
                stats["replacements"] += 1
            dnf, _ = ds_algorithm(S, ("y", 1), trace=trace)
            assert (dnf_bitset(dnf, S.terminals) == val0[("y", 1)]).all()
        # D-алгоритм на всей подсистеме (в т.ч. с циклами)
        dD = d_algorithm(derive_subsystem(S, ("y", 1)), ("y", 1))
        assert (dnf_bitset(dD, S.terminals) == val0[("y", 1)]).all()
    stats["seeds_tried"] = seed
    stats["max_height_acyclic_Sz"] = hmax
    return dict(stats)


def tau(S, a, memo):
    """tau(a) -- число деревьев вывода литерала a."""
    if S.is_term(a):
        return 1
    if a not in memo:
        tot = 0
        for L, c in S.prods:
            if c == a:
                p = 1
                for b in L:
                    p *= tau(S, b, memo)
                tot += p
        memo[a] = tot
    return memo[a]


def part_b():
    lits, inst = 0, 0
    for k in range(300):
        rng = np.random.default_rng(50000 + k)
        n = int(rng.integers(8, 13)); H = int(rng.integers(2, 5))
        S = tiered_kb(rng, n, H)
        Sz, layers, Hz, status = lael(S, ("y", 1))
        if status != "ok":
            continue
        inst += 1
        val0 = odv_bitsets(S)
        dnf, D_size = ds_algorithm(S, ("y", 1))
        assert (dnf_bitset(dnf, S.terminals) == val0[("y", 1)]).all()
        hs = heights(Sz)
        memo = {}
        tau_z = tau(Sz, ("y", 1), memo)
        kv = max(sum(1 for L, c in Sz.prods if c == a) for a in Sz.nonterminals())
        ka = max(sum(1 for b in L if not Sz.is_term(b)) for L, a in Sz.prods)
        for x, Dx in D_size.items():
            G = sum(np.prod([D_size[b] for b in L if not Sz.is_term(b)] or [1])
                    for L, a in Sz.prods if a == x)
            h = hs[x]
            mu = kv ** (h if ka == 1 else (ka ** h - 1) // (ka - 1))
            tau_x = tau(Sz, x, memo)
            assert Dx <= G <= tau_x <= tau_z and G <= mu, (k, x)
            lits += 1
    exact = []
    for kv, ka, H in [(2, 1, 4), (2, 2, 3), (3, 2, 2), (2, 3, 3), (3, 1, 5)]:
        S = tree(kv, ka, H)
        dnf, _ = ds_algorithm(S, ("y", 1))
        mu = kv ** (H if ka == 1 else (ka ** H - 1) // (ka - 1))
        assert len(dnf) == mu
        exact.append((kv, ka, H, mu))
    dd = []
    for H in range(2, 11):
        S = dvsds(H)
        z = ("b%d" % H, 1)
        st1, st2 = DStats(), DStats()
        d_algorithm(derive_subsystem(S, z), z, st1)
        ds_algorithm(S, z, st2)
        assert st1.completed == 2 ** (H - 1) and st2.completed == 2 * H - 1, (H, st1.completed, st2.completed)
        dd.append((H, st1.completed, st2.completed))
    return dict(acyclic_instances=inst, literals_checked=lits, tree_exact=exact, d_vs_ds=dd)


if __name__ == "__main__":
    print("B:", part_b(), flush=True)
    print("A:", part_a(), flush=True)
