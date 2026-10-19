"""Детерминированные семейства ЛПС для рис. 2 и утверждения 4."""
from lps import LPS

def chain(w):
    """a_1 <- x_1;  a_{i+1} <- a_i & x_{i+1} (i<w);  y <- a_w.  nu=H=q=w+1."""
    P = [((("x1", 1),), ("a1", 1))]
    for i in range(1, w):
        P.append(((("a%d" % i, 1), ("x%d" % (i + 1), 1)), ("a%d" % (i + 1), 1)))
    P.append(((("a%d" % w, 1),), ("y", 1)))
    return LPS(P, ["x%d" % i for i in range(1, w + 1)])

def wide(w):
    """a_i <- x_i;  y <- a_i  (i=1..w).  nu=2w, H=2."""
    P = [((("x%d" % i, 1),), ("a%d" % i, 1)) for i in range(1, w + 1)]
    P += [((("a%d" % i, 1),), ("y", 1)) for i in range(1, w + 1)]
    return LPS(P, ["x%d" % i for i in range(1, w + 1)])

def tree(kv, ka, H):
    """Утверждение 4 п.3: у литерала высоты h>=2 ровно kv продукций из ka
    новых посылок высоты h-1; у литерала высоты 1 -- kv продукций из одного
    нового терминального литерала."""
    P, terms, cnt = [], [], [0]
    def new(prefix):
        cnt[0] += 1
        return "%s%d" % (prefix, cnt[0])
    def build(lit, h):
        for _ in range(kv):
            if h == 1:
                t = new("x"); terms.append(t)
                P.append((((t, 1),), lit))
            else:
                ch = [(new("b"), 1) for _ in range(ka)]
                P.append((tuple(ch), lit))
                for c in ch:
                    build(c, h - 1)
    build(("y", 1), H)
    return LPS(P, terms)

def dvsds(H):
    """Утверждение 4 п.4: b_1<-x_1; b_{i+1}<-b_i; b_{i+1}<-b_i&x_{i+1}."""
    P = [((("x1", 1),), ("b1", 1))]
    for i in range(1, H):
        P.append(((("b%d" % i, 1),), ("b%d" % (i + 1), 1)))
        P.append(((("b%d" % i, 1), ("x%d" % (i + 1), 1)), ("b%d" % (i + 1), 1)))
    return LPS(P, ["x%d" % i for i in range(1, H + 1)])
