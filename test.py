import numpy as np
a = np.array((1,2))
print(a.reshape(-1))
b = np.array((3,4))
print(b.reshape(-1))
c = [a,b]
c = np.stack(c,-1)
print(c)