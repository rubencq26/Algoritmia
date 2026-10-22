import numpy as np

datos = np.random.randint(0, 100, 30)

def shell_sort(A):
    n = len(A)
    h = 1
    while h < n // 3:
        h = 3 * h + 1

    while h >= 1:
        for i in range(h, n):
            key = A[i]
            j = i - h
            while j >= 0 and A[j] > key:
                A[j + h] = A[j]
                j -= h
            A[j + h] = key
        h //= 3
    return A 

print(shell_sort(datos))