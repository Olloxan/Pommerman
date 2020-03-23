import numpy as np


a = np.arange(9)
a[0] = 4
a = len([entry for entry in a if entry == 4])
print(a)