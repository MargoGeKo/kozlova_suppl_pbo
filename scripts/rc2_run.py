"""Запуск RC2 (PySAT) в конфигурации MaxSAT Evaluation: стратификация
(BLO), adapt, exhaust, minz; SAT-оракул Glucose 4.2.1.
Печатает 'cost <стоимость>' или 'UNSAT'."""
import sys
from pysat.examples.rc2 import RC2Stratified
from pysat.formula import WCNF
w = WCNF(from_file=sys.argv[1])
with RC2Stratified(w, solver="g42", adapt=True, exhaust=True, minz=True, blo="div") as r:
    m = r.compute()
    print("UNSAT" if m is None else "cost %d" % r.cost)
