from math import *

x = 0.2
y = 0.3
r2 = x * x + y * y
B_magn = 1.3

print("Bx: ", -y * B_magn / r2)
print("By: ", x * B_magn / r2)
print("Bz: ", 0.0)
