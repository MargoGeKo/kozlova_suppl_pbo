"""Сокращение ДНФ склеиванием и поглощением (до неподвижной точки)."""

def glue_absorb(dnf):
    S = {frozenset(K) for K in dnf}
    changed = True
    while changed:
        changed = False
        L = list(S)
        new = set()
        for i in range(len(L)):
            for j in range(i + 1, len(L)):
                a, b = L[i], L[j]
                if len(a) == len(b):
                    d = a ^ b
                    if len(d) == 2:
                        (v1, s1), (v2, s2) = sorted(d)
                        if v1 == v2:
                            new.add(a & b)
        if not new <= S:
            S |= new
            changed = True
        mins = {K for K in S if not any(o < K for o in S)}
        if mins != S:
            S = mins
            changed = True
    return sorted(S, key=lambda K: (len(K), sorted(K)))
