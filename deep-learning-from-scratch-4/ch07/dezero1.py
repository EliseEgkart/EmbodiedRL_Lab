"""Basic matrix multiplication examples with DeZero Variables."""

import numpy as np
from dezero import Variable
import dezero.functions as F

# Inner product between two vectors:
# [1, 2, 3] dot [4, 5, 6] = 1*4 + 2*5 + 3*6
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
a, b = Variable(a), Variable(b)
c = F.matmul(a, b)
print(c)

# Matrix product:
# This is the linear algebra primitive underlying affine layers.
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
c = F.matmul(a, b)
print(c)
