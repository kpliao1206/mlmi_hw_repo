#!/usr/bin/env python3

###  https://crypto.stanford.edu/pbc/notes/pi/ramanujan.html

import math
import time

# 開始測量
start_time = time.time()
print('start at ' + time.strftime('%H:%M:%S'))

def factorial(n):
    if n==0:
        return 1
    else:
        return n*factorial(n-1)

def pi():
    sum =0
    k=0
    f=2*(math.sqrt(2))/9801
    while True:
        fz = (26390*k + 1103)*factorial(4*k)
        fm = (396**(4*k))*((factorial(k))**4)
        t = f*fz/fm
        sum += t
        if t<1e-15:
            break
        k += 1
    return 1/sum

print("calcuted pi is：",pi())

print("real pi is：",math.pi)

print('stop at ' + time.strftime('%H:%M:%S'))

end_time = time.time()

# %-formatting
print("execution time: %f sec." % (end_time - start_time))

# str.format (Format Specification Mini-Language)
print('execution time: {et} sec.'.format(et=end_time - start_time))

#  f-string (Literal String Interpolation)
print(f'execution time: {end_time - start_time} sec.')