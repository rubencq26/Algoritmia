DATA = [1, 2, 3, 4, 5 ,6 ,7, 8, 9]

def  busqueda_binaria(d, x):
    lo, hi = 0, len(d) - 1
    while lo <= hi:
        medio = lo + (hi - lo) // 2
        if d[medio] == x:
            return medio
        if d[medio] < x:
            lo = medio + 1
        else:
            hi = medio - 1
    return -1
        

print(busqueda_binaria(DATA, 2))
print(busqueda_binaria(DATA, 10))
    
    
