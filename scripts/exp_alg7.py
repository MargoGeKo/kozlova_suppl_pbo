"""Сравнение стратегий выбора запроса (разд. 3): число запросов в худшем
случае (ответы выбирает противник из совместных).

Группа 1 (24 задачи): зёрна 70000+k; n ~ U{6..10}, m ~ U{3..8},
ДНФ random_dnf (ранг ~ U{1..n//2}), знаки коэффициентов +-1
равновероятно; задача принимается, если M' = 4 (первые 12) или M' = 5
(первые 12). Для неё вычисляется и оптимум уравнения Беллмана.
Группа 2 (36 задач): product_classes с размерами блоков M'_t = (2,2), (2,3),
(2,2,2), (3,3) -- по 9 задач, зёрна 80000+100*b+k.
Последовательное сравнение: порядок классов -- случайная перестановка
(зерно задачи + 777)."""
import csv, json, sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from alg7 import (LP_CALLS, Problem, build_classes, make_sequential, optimal,
                  strategy_css, strategy_g, worst_case)
from generators import product_classes, random_dnf

rows = []
tasks = open(ROOT / "data/alg7_tasks.jsonl", "w")


def run(group, seed, n, D, sg, with_opt):
    cl, Ppm, dnf_Pi = build_classes(n, D, sg)
    prob = Problem(cl)
    order = list(np.random.default_rng(seed + 777).permutation(prob.Mprime))
    t0 = time.time()
    rec = dict(group=group, seed=seed, n=n, m=len(D), Mprime=prob.Mprime,
               g=worst_case(prob, strategy_g),
               seq=worst_case(prob, make_sequential([int(a) for a in order])),
               css=worst_case(prob, strategy_css))
    d0 = strategy_g(prob, ()); kap = prob.kappa(())
    rec["g0_rule"] = kap - max(prob.kappa(((d0, r),)) for r in (">", "<", "=") if prob.consistent(((d0, r),)))
    d1 = strategy_css(prob, ())
    R1 = [r for r in (">", "<", "=") if prob.consistent(((d1, r),))]
    rec["g0_css"] = kap - max(prob.kappa(((d1, r),)) for r in R1)
    rec["opt"] = optimal(prob) if with_opt else ""
    rec["log2"] = int(np.ceil(np.log2(prob.Mprime)))
    rec["sec"] = round(time.time() - t0, 1)
    rows.append(rec)
    tasks.write(json.dumps(dict(group=group, seed=seed, n=n, D=D, signs=sg)) + "\n"); tasks.flush()
    print(rec, flush=True)


need = {4: 12, 5: 12}
k = 0
while any(need.values()):
    seed = 70000 + k; k += 1
    rng = np.random.default_rng(seed)
    n = int(rng.integers(6, 11)); m = int(rng.integers(3, 9))
    D = random_dnf(rng, n, m)
    sg = [int(rng.choice([1, -1])) for _ in range(n)]
    cl, _, _ = build_classes(n, D, sg)
    if need.get(len(cl), 0) > 0:
        need[len(cl)] -= 1
        run("1", seed, n, D, sg, True)

for b, block_sizes in enumerate([(2, 2), (2, 3), (2, 2, 2), (3, 3)]):
    for kk in range(9):
        seed = 80000 + 100 * b + kk
        n, D, sg = product_classes(np.random.default_rng(seed), list(block_sizes))
        run("2", seed, n, D, sg, False)

with open(ROOT / "results/alg7_rows.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print("LP calls:", LP_CALLS[0])
