import numpy as np
import random

a = np.array([1,2,3])
b = np.array([4,5,6])
c = np.array([7,8,9])
d = np.array([10,11,12])

mem = []

mem.append(a)
print(a.shape)
print(mem)
mem.append(b)
print(mem)
mem.append(c)
print(mem)
mem.append(d)
print(mem)

res = np.array(random.sample(mem, 3))
print(res.shape)
print(type(res))
print(res)


ddddd = np.vstack(res)
print(ddddd)


