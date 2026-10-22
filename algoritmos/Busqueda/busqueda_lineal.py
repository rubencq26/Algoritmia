DATA = [6, 5, 3, 2, 0, 9]

def busqueda_lineal(d, x):
    for i in range(len(d)):
        if(d[i] == x):
            return i
    return -1


print(busqueda_lineal(DATA, 2))
print(busqueda_lineal(DATA, 7))