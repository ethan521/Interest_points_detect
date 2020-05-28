#coding=utf-8

import numpy as np


a = np.arange(0,8)
print(a)

b= a.reshape((4,2))
print(b)
print(np.transpose(b))
