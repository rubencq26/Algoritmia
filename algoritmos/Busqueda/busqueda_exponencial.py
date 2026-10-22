
DATA = range(100000)

def busqueda_exponencial(d, x):
    power = 0

    while 2 ** power < x:
        power += 1

    lo, hi = 2 ** power // 2, 2 ** power

    while lo <= hi:
        medio = lo + (hi - lo) // 2

        if d[medio] == x:
            return medio
        if d[medio] < x:
            lo = medio + 1
        else:
            hi = medio - 1
    return -1

print(busqueda_exponencial(DATA, 7428))
print(busqueda_exponencial(DATA, -1))



