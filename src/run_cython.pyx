# Tail Recursive Factorial

cpdef int fact_tailrecursive(int n, int a = 1):
 
    if (n == 0):
        return a
 
    return fact_tailrecursive(n - 1, n * a)


# Loop Factorial

cpdef int loop_factor(int x):
    cdef int y = 1
    cdef int i
    for i in range(1, x+1):
        y *= i
    return y

