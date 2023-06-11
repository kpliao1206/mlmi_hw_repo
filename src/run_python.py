import math

# Tail Recursive Factorial

def fact_tailrecursive(n, a = 1):
 
    if (n == 0):
        return a
 
    return fact_tailrecursive(n - 1, n * a)

# Loop Factorial

def loop_factor(x):
    y = 1
    for i in range(1, x+1):
        y *= i
    return y

#Inbuilt Factorial (Gamma)

def fact_gamma(n):
    return math.gamma(n + 1)