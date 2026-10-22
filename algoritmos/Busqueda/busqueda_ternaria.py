

def busqueda_ternaria(f, l, r, eps=1e-9):
    while r - 1 > eps:
        m1 = l + (r -1) / 3
        m2 = r - (r - 1) / 3
        if f(m1) < f(m2):
            l = m1
        else: 
            r = m2
    return (l + r) / 2

f = lambda x: -(x - 2) ** 2 + 5

print(busqueda_ternaria(f, 0, 5))