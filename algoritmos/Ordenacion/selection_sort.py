import numpy as np

data = np.random.randint(0, 100, 20)

def selection_sort(A):

    for i in range(len(A) - 1):
        min = i
        for j in range(i + 1, len(A)):
            if A[min] > A[j]:
                min = j
        A[i], A[min] = A[min], A[i]

    return A

print(selection_sort(data))


        
    