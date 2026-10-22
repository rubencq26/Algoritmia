import numpy as np

datos = np.random.randint(0, 100, 30)

def bubble_sort(d):
    for i in range(len(d) - 1):
        swapped =  False
        for j in range(len(d) - 1 -i):
            if d[j] > d[j + 1]:
                d[j], d[j + 1] = d[j + 1], d[j]
                swapped = True
        if not swapped:
            return(d)

print(bubble_sort(datos))


            
