import torch
import numpy as np

a = []
a.append([1])
a.append([2])
a.append([3])
print(a)
print(np.shape(a))

a = np.resize(a, (len(a),))
print(a)
print(np.shape(a))


a = np.asarray(a)
print(a)
print(np.shape(a))

