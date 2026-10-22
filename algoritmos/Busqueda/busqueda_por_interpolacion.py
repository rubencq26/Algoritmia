DATA = [1, 2, 3, 4, 5 ,6 ,7, 8, 9]

def busqueda_interpolacion(d, x):
    lo, hi= 0, len(d) -1

    while lo <= hi and d[lo] <= x <= d[hi]:
        if d[lo] == d[hi]:
            return lo

        pos = lo + (x - d[lo] * (hi - lo)) // (d[hi] - d[lo])

        if d[pos] == x:
            return pos
        if d[pos] < x:
            lo = pos + 1
        else: 
            hi = pos - 1
    return -1

print(busqueda_interpolacion(DATA, 2))
print(busqueda_interpolacion(DATA, 10))