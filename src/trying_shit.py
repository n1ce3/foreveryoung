import torch.nn as nn

layer = nn.Conv2d(1, 32, 2, 2)

kernel_sizes = [(1, 32, 4, 2), (32, 64, 4, 2)]

# initialize convolutional layers
layer_list = []
for shape in kernel_sizes[:-1]:
    print(shape[3])
