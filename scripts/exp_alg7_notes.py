"""Замечание о классах (M' <= m, построение без перебора Delta) и обработка
противоречивых ответов на примере 5. Выход: results/alg7_notes.txt."""
import itertools, json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from alg7 import Problem, build_PiA, build_classes, conflict_set, max_consistent
from generators import random_dnf

out = []
def points(n, dnf_Pi):
    res = set()
    for cl in dnf_Pi.values():
        for K in cl:
            fix = dict(K); free = [i for i in range(n) if i not in fix]
            for b in itertools.product((0, 1), repeat=len(free)):
                x = dict(fix); x.update(zip(free, b)); res.add(tuple(x[i] for i in range(n)))
    return res

cnt = 0; ratio = 0
for t in map(json.loads, open(ROOT / "data/alg7_tasks.jsonl")):
    D = [tuple(tuple(l) for l in K) for K in t["D"]]
    c1, Pi, _ = build_PiA(t["n"], D, t["signs"]); c2, _, dp = build_classes(t["n"], D, t["signs"])
    assert c1 == c2 and set(Pi) == points(t["n"], dp); cnt += 1
rng = np.random.default_rng(123)
for k in range(3000):
    n = int(rng.integers(3, 11)); m = int(rng.integers(1, 12))
    D = random_dnf(rng, n, m); sg = [int(v) for v in rng.choice([-1, 0, 1], size=n, p=[.4, .2, .4])]
    c1, Pi, _ = build_PiA(n, D, sg); c2, _, dp = build_classes(n, D, sg)
    assert c1 == c2 and set(Pi) == points(n, dp) and len(c2) <= m; cnt += 1
out.append("классы без перебора Delta совпали с явным перебором: %d задач (60 + 3000 случайных, "
           "n=3..10, m=1..11, P(c_i=0)=0.2); M' <= m во всех" % cnt)

# пример 5
D = [((0, 0), (2, 0)), ((3, 1), (4, 1)), ((2, 0), (3, 0), (4, 1)), ((0, 0), (1, 1), (3, 1))]
sg = [1, 1, 1, -1, -1]
cl, Ppm, dnf_Pi = build_classes(5, D, sg); prob = Problem(cl)
y = lambda x: tuple(int(x[i]) if sg[i] > 0 else 1 - int(x[i]) for i in Ppm)
d = lambda a, b: tuple(p - q for p, q in zip(y(a), y(b)))
a1 = (d('01110', '01000'), '>')   # c3 > |c4|
a2 = (d('11001', '01000'), '>')   # c1 > |c5|
q = d('11111', '01000')           # контрольный: c1 + c3 > |c4| + |c5| ?
R = [r for r in '><=' if prob.consistent((a1, a2, (q, r)))]
out.append("пример 5: после ответов 1-2 совместные ответы на контрольный запрос: %s" % R)
ans = [a1, a2, (q, '<')]
out.append("ответ «нет»: система совместна? %s; минимальный противоречивый набор: %s; "
           "наибольший совместный: %s" % (prob.consistent(tuple(ans)), conflict_set(prob, ans),
                                         max_consistent(prob, ans)))
open(ROOT / "results/alg7_notes.txt", "w").write("\n".join(out) + "\n")
print("\n".join(out))
