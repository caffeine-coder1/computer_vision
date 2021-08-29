# import torch

# a = torch.rand([10, 1, 128])
# b, c = torch.min(a, -1)
# print(torch.typename(b))
# print(b)
# print(c)
# bol = b < 0.01
# print(bol)
# d = torch.where(bol, c, 43)

# print(d)

import numpy as np

a = np.random.randint(0, 100, 25)
print(a)
b = np.random.choice(a, 1)
print(b)
