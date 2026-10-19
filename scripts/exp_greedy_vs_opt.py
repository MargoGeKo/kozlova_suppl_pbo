"""Правило (6) против оптимальной стратегии (разд. 3, «Вычислительная
проверка»): 60 задач алгоритма 7 и 115 задач G6 с M' <= 9.
Для каждой задачи: g -- худший случай правила (6) (ничьи -- по порядку),
g_ties -- при наихудшем разрешении оставшихся ничьих, opt -- оптимум
уравнения Беллмана на попарных запросах (считается, только если g_ties
больше нижней оценки ceil(log2 M')). Выход: results/greedy_vs_opt.csv."""
import csv, json, math, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from alg7 import Problem, build_classes, optimal, strategy_g, worst_case, worst_case_ties
from generators import product_variants

insts = []
for t in map(json.loads, open(ROOT / "data/alg7_tasks.jsonl")):
    D = [tuple(tuple(l) for l in K) for K in t["D"]]
    insts.append(("alg7_group%s_%d" % (t["group"], t["seed"]), build_classes(t["n"], D, t["signs"])[0]))
for k, (name, Y) in enumerate(product_variants()):
    if len(Y) <= 9:
        insts.append(("G6_%s_%d" % (name, k), Y))
rows = []
for name, Y in insts:
    t0 = time.time()
    prob = Problem([tuple(y) for y in Y])
    lb = math.ceil(math.log2(prob.Mprime))
    g = worst_case(prob, strategy_g)
    gt = worst_case_ties(prob)
    opt = optimal(prob) if gt > lb else lb
    rows.append(dict(name=name, Mprime=prob.Mprime, lower=lb, g=g, g_ties=gt, opt=opt,
                     sec=round(time.time() - t0, 1)))
    print(rows[-1], flush=True)
with open(ROOT / "results/greedy_vs_opt.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
lbm = lambda r: r["lower"] < r["Mprime"] - 1
print("задач:", len(rows), "; правило хуже оптимума:", sum(r["g_ties"] > r["opt"] for r in rows),
      "; opt=нижняя<M'-1:", sum(r["opt"] == r["lower"] and lbm(r) for r in rows),
      "; opt=M'-1>нижней:", sum(r["opt"] == r["Mprime"] - 1 and lbm(r) for r in rows),
      "; строго между:", sum(r["lower"] < r["opt"] < r["Mprime"] - 1 for r in rows),
      "; оценки совпадают:", sum(not lbm(r) for r in rows))
