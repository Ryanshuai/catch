import numpy as np

a = np.arange(160).reshape(32,5)
print(a)

b = a*np.array([1,1,1,30,30]).reshape(1,5)
print(b)