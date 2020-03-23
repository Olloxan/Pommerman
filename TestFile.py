import numpy as np




test = np.ones((10,10,1))
test2 = np.ones((10,10,1))

test[1,1] = 2

test3 = test - test2 

print(test3)