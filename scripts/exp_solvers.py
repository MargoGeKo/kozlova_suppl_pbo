"""Сравнение с современными решателями псевдобулевых задач.

Семейство DNF («ДНФ», каноническая модель): max f при D=1, случайная ДНФ.
  n in {100, 1000}, m in {100, 1000, 10000}, ранг конъюнкции ~ U{5..20},
  c_i ~ U{-100..100}; 5 задач на ячейку, зёрна 90000+1000*[n=1000]+10*im+k.
  Метод статьи: формула (2), O(mn).
Семейство KB («БЗ», база знаний): ярусная ЛПС generators.tiered_kb с n in {50,
  100, 200}, H in {4, 6, 8}, 3..6 фактов в ярусе (остальные параметры как в
  табл. 6), c_i ~ U{-100..100}; 5 задач на ячейку, зёрна 95000+...
  Метод статьи: DS-алгоритм + формула (2) (лимит 60 с).
Решатели получают одну и ту же клаузальную модель (encode.py): HiGHS -- LP,
SCIP и CP-SAT -- OPB. Лимит 60 с, 1 поток. Семейство KB-large: три крупные ЛПС (n=1000, 5000, 20000; H=10, 15, 20;
  20..40, 100..200, 400..800 фактов в ярусе) -- только решатели.
Выход: results/solvers_rows.csv (DNF, KB), results/solvers_large_rows.csv (KB-large).
Столбец D_size -- |D| (число конъюнкций ДНФ; для KB -- построенной D_y)."""
import csv
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from encode import dnf_model, lps_model  # noqa
from generators import random_dnf, tiered_kb  # noqa
from lps import ds_algorithm, lit_str  # noqa

BIN = json.load(open(ROOT / "scripts/solvers.json"))
TL = 60


class Timeout(Exception):
    pass


def _alarm(sig, frm):
    raise Timeout()


def formula(n, D, c):
    """Формула (2): максимум на каждом интервале, лучший из них."""
    best, arg = None, None
    pos = sum(v for v in c if v > 0)
    for j, K in enumerate(D):
        val = pos
        for i, s in K:
            if c[i] > 0 and s == 0:
                val -= c[i]
            elif c[i] < 0 and s == 1:
                val += c[i]
        if best is None or val > best:
            best, arg = val, j
    return best, arg


def run_highs(M, tmp):
    f = os.path.join(tmp, "m.lp"); M.to_lp(f)
    t0 = time.perf_counter()
    p = subprocess.run([BIN["highs"], "--model_file", f, "--time_limit", str(TL), "--threads", "1",
                        "--random_seed", "0"], capture_output=True, text=True)
    t = time.perf_counter() - t0
    st = re.search(r"^\s*Status\s+(\S+)", p.stdout, re.M) or \
        re.search(r"Model status\s*:\s*(\S+)", p.stdout)
    ob = re.search(r"^\s*Primal bound\s+(\S+)", p.stdout, re.M) or \
        re.search(r"Objective value\s*:\s*(\S+)", p.stdout)
    ok = st and st.group(1).strip() == "Optimal"
    return (round(float(ob.group(1))) if ok else None), t


def run_scip(M, tmp):
    f = os.path.join(tmp, "m.opb"); M.to_opb(f)
    t0 = time.perf_counter()
    p = subprocess.run([BIN["scip"], "-c", "set limits time %d" % TL, "-c", "set parallel maxnthreads 1",
                        "-c", "read %s" % f, "-c", "optimize", "-c", "quit"],
                       capture_output=True, text=True)
    t = time.perf_counter() - t0
    ok = "optimal solution found" in p.stdout
    ob = re.search(r"Primal Bound\s*:\s*(\S+)", p.stdout)
    return (-round(float(ob.group(1))) if ok and ob else None), t


def run_cpsat(M, tmp):
    f = os.path.join(tmp, "m.opb"); M.to_opb(f)
    t0 = time.perf_counter()
    p = subprocess.run([BIN["cpsat"], "--input=%s" % f,
                        "--params=num_workers:1,max_time_in_seconds:%d,random_seed:0" % TL],
                       capture_output=True, text=True)
    t = time.perf_counter() - t0
    out = p.stdout + p.stderr
    ok = re.search(r"status:\s*OPTIMAL", out)
    ob = re.search(r"^objective:\s*(\S+)", out, re.M)
    return (-round(float(ob.group(1))) if ok and ob else None), t


def run_rc2(M, tmp):
    f = os.path.join(tmp, "m.wcnf"); posw = M.to_wcnf(f)
    env = dict(os.environ, PYTHONPATH=BIN["pysat"])
    t0 = time.perf_counter()
    try:
        p = subprocess.run([sys.executable, str(ROOT / "scripts/rc2_run.py"), f], capture_output=True,
                           text=True, timeout=TL, env=env)
    except subprocess.TimeoutExpired:
        return None, time.perf_counter() - t0
    t = time.perf_counter() - t0
    mm = re.search(r"cost (\d+)", p.stdout)
    return (posw - int(mm.group(1)) if mm else None), t


def make_pbcomp(cmd):
    """Решатель в формате PB Competition (RoundingSat, Exact, ...):
    cmd -- шаблон команды с {file}; ожидаются строки 'o <val>' и
    's OPTIMUM FOUND'."""
    def run(M, tmp):
        f = os.path.join(tmp, "m.opb"); M.to_opb(f)
        t0 = time.perf_counter()
        try:
            p = subprocess.run(cmd.format(file=f).split(), capture_output=True, text=True, timeout=TL)
        except subprocess.TimeoutExpired:
            return None, time.perf_counter() - t0
        t = time.perf_counter() - t0
        ok = re.search(r"^s OPTIMUM FOUND", p.stdout, re.M)
        os_ = re.findall(r"^o\s+(-?\d+)", p.stdout, re.M)
        return (-int(os_[-1]) if ok and os_ else None), t
    return run


def _available(key):
    path = BIN.get(key)
    if not path:
        return False
    if key == "pysat":
        env = dict(os.environ, PYTHONPATH=path)
        return subprocess.run([sys.executable, "-c", "import pysat.examples.rc2"], env=env,
                              capture_output=True).returncode == 0
    return bool(shutil.which(path))


SOLVERS = {k: v for k, v in [("HiGHS", run_highs), ("SCIP", run_scip), ("CP-SAT", run_cpsat),
                             ("RC2", run_rc2)]
           if _available({"HiGHS": "highs", "SCIP": "scip", "CP-SAT": "cpsat", "RC2": "pysat"}[k])}
# дополнительные решатели, например {"RoundingSat": "roundingsat {file}"}
for _name, _cmd in BIN.get("pbcomp", {}).items():
    SOLVERS[_name] = make_pbcomp(_cmd)


def bench(rows, fam, cell, seed, ours, model):
    rec = dict(family=fam, **cell, seed=seed, vars=len(model.names), clauses=len(model.clauses))
    rec.update(ours)
    with tempfile.TemporaryDirectory() as tmp:
        for name, fn in SOLVERS.items():
            v, t = fn(model, tmp)
            rec[name + "_val"] = v
            rec[name + "_t"] = round(t, 3)
    rows.append(rec)
    print(rec, flush=True)


def main(families=("DNF", "KB")):
    rows = []
    signal.signal(signal.SIGALRM, _alarm)
    if "DNF" in families:
        for big, n in enumerate((100, 1000)):
            for im, m in enumerate((100, 1000, 10000)):
                for k in range(5):
                    seed = 90000 + 1000 * big + 10 * im + k
                    rng = np.random.default_rng(seed)
                    D = random_dnf(rng, n, m, rank=(5, 20))
                    c = [int(v) for v in rng.integers(-100, 101, size=n)]
                    t0 = time.perf_counter(); val, j = formula(n, D, c)
                    t = time.perf_counter() - t0
                    bench(rows, "DNF", dict(n=n, m=m), seed,
                          dict(ours_val=val, ours_t=round(t, 4), D_size=m), dnf_model(n, D, c))
    if "KB" in families:
        for a, n in enumerate((50, 100, 200)):
            for b, H in enumerate((4, 6, 8)):
                for k in range(5):
                    seed = 95000 + 100 * a + 10 * b + k
                    rng = np.random.default_rng(seed)
                    S = tiered_kb(rng, n, H, facts=(3, 6))
                    c = [int(v) for v in rng.integers(-100, 101, size=n)]
                    cd = dict(zip(S.terminals, c))
                    signal.alarm(TL)
                    t0 = time.perf_counter()
                    try:
                        dnf, _ = ds_algorithm(S, ("y", 1))
                        signal.alarm(0)
                        pos = {v: i for i, v in enumerate(S.terminals)}
                        D = [tuple((pos[v], s) for v, s in K) for K in dnf]
                        val = formula(n, D, c)[0] if D else None
                        ours = dict(ours_val=val, ours_t=round(time.perf_counter() - t0, 4), D_size=len(dnf))
                    except Timeout:
                        ours = dict(ours_val=None, ours_t=None, D_size=None)
                    bench(rows, "KB", dict(n=n, H=H), seed, ours, lps_model(S, cd))
    if "KB-large" in families:
        # крупные ЛПС: только решатели (DS-синтез не выполняется)
        for n, H, f in ((1000, 10, (20, 40)), (5000, 15, (100, 200)), (20000, 20, (400, 800))):
            seed = 97000 + n
            rng = np.random.default_rng(seed)
            S = tiered_kb(rng, n, H, facts=f)
            c = [int(v) for v in rng.integers(-100, 101, size=n)]
            bench(rows, "KB-large", dict(n=n, H=H), seed, dict(ours_val=None, ours_t=None, D_size=None),
                  lps_model(S, dict(zip(S.terminals, c))))
    keys = sorted({k for r in rows for k in r}, key=lambda k: list(rows[-1]).index(k) if k in rows[-1] else 99)
    name = "solvers_rows.csv" if "KB-large" not in families else "solvers_large_rows.csv"
    with open(ROOT / "results" / name, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main(tuple(sys.argv[1:]) or ("DNF", "KB"))
