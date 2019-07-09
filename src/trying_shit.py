import torch.nn as nn
from datetime import datetime
from numpy.random import normal
import numpy as np
layer = nn.Conv2d(1, 32, 2, 2)

kernel_sizes = [(1, 32, 4, 2), (32, 64, 4, 2)]

# initialize convolutional layers
layer_list = []
print(len(np.logspace(-3, -4, 10)))

a = [0.001]
print(len(a))
print(len(np.array(a)))
