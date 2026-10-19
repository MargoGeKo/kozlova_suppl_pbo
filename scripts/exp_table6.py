"""Эксперимент разд. 6 (табл. 6): ДНФ-описания на случайных ярусных ЛПС.

Для каждой пары (n, H) перебираются зёрна seed = 1000*n + 100*H + k,
k = 0, 1, ..., пока не набрано 20 ЛПС с непустой ОДВ(y).
Выход: data/table6_kb.jsonl (ЛПС), results/table6_rows.csv (по ЛПС),
results/table6.csv (медианы/средние)."""
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lps import (DStats, d_algorithm, derive_subsystem, dnf_bitset,  # noqa
                 ds_algorithm, lit_str, odv_bitsets)
from generators import tiered_kb  # noqa

PER_CELL, N_PROBES = 20, 30


def popcount(a):
    return np.unpackbits(a.view(np.uint8)).reshape(len(a), -1).sum(1)


def run():
    rows = []
    kbf = open(ROOT / "data/table6_kb.jsonl", "w")
    for n in (12, 16, 20):
        idx = np.arange(1 << n, dtype=np.int64)
        for H in (2, 3, 4):
            got, k = 0, 0
            while got < PER_CELL:
                seed = 1000 * n + 100 * H + k
                k += 1
                rng = np.random.default_rng(seed)
                S = tiered_kb(rng, n, H)
                val = odv_bitsets(S)
                omega = val[("y", 1)]
                if not omega.any():
                    continue
                got += 1
                Sz = derive_subsystem(S, ("y", 1))
                stD, stDS = DStats(), DStats()
                t0 = time.perf_counter()
                dnfD = d_algorithm(Sz, ("y", 1), stD)
                tD = time.perf_counter() - t0
                t0 = time.perf_counter()
                dnf, D_size = ds_algorithm(S, ("y", 1), stDS)
                tDS = time.perf_counter() - t0
                assert (dnf_bitset(dnf, S.terminals) == omega).all()
                assert (dnf_bitset(dnfD, S.terminals) == omega).all()
                pos = {v: i for i, v in enumerate(S.terminals)}
                cost = rng.integers(5, 101, size=n)
                costs = [sum(int(cost[pos[v]]) for v, s in K if s) for K in dnf]
                jbest = int(np.argmin(costs))
                # проверка: минимум стоимости по ДНФ = минимум перебором
                tot = np.zeros(1 << n, dtype=np.int64)
                for i in range(n):
                    tot += ((idx >> i) & 1) * int(cost[i])
                assert costs[jbest] == tot[omega].min()
                # наименьшая правка
                infe = np.flatnonzero(~omega)
                probes = rng.choice(infe, size=N_PROBES, replace=len(infe) < N_PROBES)
                fe = np.flatnonzero(omega)
                ds_ = []
                for p in probes:
                    bits = {v: (int(p) >> pos[v]) & 1 for v in S.terminals}
                    d = min(sum(bits[v] != s for v, s in K) for K in dnf)
                    brute = popcount(np.bitwise_xor(fe, p)).min()
                    assert d == brute, (d, brute)
                    ds_.append(d)
                rec = dict(n=n, H=H, seed=seed, nu=S.nu, D_size=len(dnf),
                           rbar=float(np.mean([len(K) for K in dnf])),
                           omega=int(omega.sum()),
                           rstar=len(dnf[jbest]), dbar=float(np.mean(ds_)),
                           D_written=stD.written, DS_written=stDS.written,
                           ratio=stDS.written / stD.written,
                           sum_D_inter=int(sum(v for a, v in D_size.items() if a != ("y", 1))),
                           t_D=tD, t_DS=tDS)
                rows.append(rec)
                kbf.write(json.dumps(dict(seed=seed, n=n, H=H,
                          terminals=S.terminals,
                          prods=[[[lit_str(b) for b in L], lit_str(a)] for L, a in S.prods],
                          costs=[int(c) for c in cost])) + "\n")
            print(n, H, "done", flush=True)
    kbf.close()
    with open(ROOT / "results/table6_rows.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    out = []
    for n in (12, 16, 20):
        for H in (2, 3, 4):
            R = [r for r in rows if r["n"] == n and r["H"] == H]
            med = lambda key: float(np.median([r[key] for r in R]))
            mean = lambda key: float(np.mean([r[key] for r in R]))
            out.append(dict(n=n, H=H, D_size=med("D_size"), rbar=mean("rbar"),
                            omega=med("omega"),
                            frac=100 * med("omega") / 2 ** n,
                            omega_per_D=float(np.median([r["omega"] / r["D_size"] for r in R])),
                            rstar=mean("rstar"), dbar=mean("dbar"),
                            ratio=med("ratio"), sum_D=med("sum_D_inter")))
    with open(ROOT / "results/table6.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader(); w.writerows(out)
    for r in out:
        print(" & ".join("%g" % (round(v, 2) if isinstance(v, float) else v) for v in r.values()))


if __name__ == "__main__":
    run()
