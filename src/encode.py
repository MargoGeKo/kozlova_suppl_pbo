"""Кодирование задач для внешних решателей.

Все задачи приводятся к виду  max sum_i c_i x_i  при системе клауз
(дизъюнкций литералов) над булевыми переменными:
  * ДНФ-ограничение D = K_1 v ... v K_m (селекторная кодировка):
        z_1 + ... + z_m >= 1,   z_j -> l  для каждого литерала l из K_j;
  * ЛПС (ацикличная, без отрицаний нетерминальных литералов):
    односторонняя полнота Кларка (кодировка Плейстеда--Гринбаума)
        y = 1;   a -> p_{j_1} v ... v p_{j_k}  (продукции L_j -> a);
        p_j -> l  для каждой посылки l из L_j.
    Наборы терминальных переменных, на которых система совместна, -- это
    в точности ОДВ(y): решение системы задаёт дерево вывода, и обратно.
Форматы: OPB (SCIP, CP-SAT, RoundingSat/Exact), LP (HiGHS), WCNF (MaxSAT).
"""
from __future__ import annotations


class ClauseModel:
    def __init__(self):
        self.names = []          # имя переменной -> индекс+1
        self.index = {}
        self.clauses = []        # списки ненулевых целых (DIMACS)
        self.obj = {}            # индекс -> коэффициент (максимизация)

    def var(self, name):
        if name not in self.index:
            self.names.append(name)
            self.index[name] = len(self.names)
        return self.index[name]

    def lit(self, name, sign):
        v = self.var(name)
        return v if sign else -v

    # ---------------------------------------------------------- writers
    def to_opb(self, path):
        with open(path, "w") as f:
            f.write("* #variable= %d #constraint= %d\n" % (len(self.names), len(self.clauses)))
            # OPB минимизирует: min -f
            terms = " ".join("%+d x%d" % (-c, v) for v, c in sorted(self.obj.items()) if c)
            f.write("min: %s ;\n" % terms)
            for cl in self.clauses:
                # sum l >= 1, где ~x = 1 - x
                lhs, rhs = [], 1
                for l in cl:
                    if l > 0:
                        lhs.append("+1 x%d" % l)
                    else:
                        lhs.append("-1 x%d" % -l); rhs -= 1
                f.write("%s >= %d ;\n" % (" ".join(lhs), rhs))

    def to_lp(self, path):
        with open(path, "w") as f:
            f.write("Maximize\n obj: ")
            terms = ["%+d x%d" % (c, v) for v, c in sorted(self.obj.items()) if c]
            f.write(" ".join(terms) if terms else "0 x1")
            f.write("\nSubject To\n")
            for k, cl in enumerate(self.clauses):
                lhs, rhs = [], 1
                for l in cl:
                    if l > 0:
                        lhs.append("+ x%d" % l)
                    else:
                        lhs.append("- x%d" % -l); rhs -= 1
                f.write(" c%d: %s >= %d\n" % (k, " ".join(lhs), rhs))
            f.write("Binaries\n")
            for v in range(1, len(self.names) + 1):
                f.write(" x%d\n" % v)
            f.write("End\n")

    def to_wcnf(self, path):
        """Старый формат WCNF: мягкие единичные клаузы с весами |c_i|.
        max f = sum_{c>0} c - (вес нарушенных мягких клауз)."""
        soft = [(c, v) for v, c in sorted(self.obj.items()) if c]
        top = sum(abs(c) for c, _ in soft) + 1
        with open(path, "w") as f:
            f.write("p wcnf %d %d %d\n" % (len(self.names), len(self.clauses) + len(soft), top))
            for cl in self.clauses:
                f.write("%d %s 0\n" % (top, " ".join(map(str, cl))))
            for c, v in soft:
                f.write("%d %d 0\n" % (abs(c), v if c > 0 else -v))
        return sum(c for c, _ in soft if c > 0)


def dnf_model(n, D, c):
    """D: список конъюнкций [(i, sigma)], c: коэффициенты (максимизация)."""
    M = ClauseModel()
    for i in range(n):
        M.var("x%d" % (i + 1))
    for i in range(n):
        if c[i]:
            M.obj[M.index["x%d" % (i + 1)]] = int(c[i])
    zs = [M.var("z%d" % (j + 1)) for j in range(len(D))]
    M.clauses.append(list(zs))
    for z, K in zip(zs, D):
        for i, s in K:
            M.clauses.append([-z, M.lit("x%d" % (i + 1), s)])
    return M


def lps_model(S, c, goal=("y", 1)):
    """S: ацикличная ЛПС (lps.LPS), c: dict терминальная переменная -> коэф."""
    M = ClauseModel()
    for v in S.terminals:
        M.var(v)
    for v, cv in c.items():
        if cv:
            M.obj[M.index[v]] = int(cv)
    nm = lambda a: ("t:" if S.is_term(a) else "n:") + a[0] + ("" if a[1] else "~")
    byc = S.by_concl()
    M.clauses.append([M.var(nm(goal))])
    for a, js in byc.items():
        ps = [M.var("p%d" % j) for j in js]
        M.clauses.append([-M.var(nm(a))] + ps)
        for j, p in zip(js, ps):
            for b in S.prods[j][0]:
                if S.is_term(b):
                    M.clauses.append([-p, M.lit(b[0], b[1])])
                else:
                    M.clauses.append([-p, M.var(nm(b))])
    # невыводимые литералы (без продукций) становятся ложными автоматически:
    # у них нет клаузы a -> ..., но и использовать их нельзя -- добавим ~a
    for a in S.nonterminals():
        if a not in byc:
            M.clauses.append([-M.var(nm(a))])
    return M
