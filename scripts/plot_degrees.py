import matplotlib.pyplot as plt
import math
import numpy as np

degrees = list(map(int, open('data/degrees.txt').readlines()))

simd_width = 8

degrees_simd = []
for i in range(min(degrees), max(degrees) + 1):
    counter = len(list(filter(lambda x: x == i, degrees)))
    actual = math.ceil(i / simd_width) * simd_width / i
    degrees_simd += [i] * int(counter * actual)
    # list(map(lambda x: , degrees))

print(sum(degrees_simd))
print(sum(degrees))
print(sum(degrees_simd)/sum(degrees))
N, bins, patches = plt.hist(degrees_simd, bins=np.arange(4.5, 33.5, 1), label="Actual bandwidth")

N, bins, patches = plt.hist(degrees, bins=np.arange(4.5, 33.5, 1), label="Neighbour degrees (Effective bandwidth)")
plt.legend()

#for i, p in enumerate(patches):
#    if i > 3:
#        p.set_facecolor('red')
plt.show()
