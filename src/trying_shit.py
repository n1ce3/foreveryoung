from datetime import datetime
from numpy.random import normal
import numpy as np
import torch
from models import VAE
from torchsummary import summary
from utils import standard_vae, newest, set_split
import glob
from PIL import Image
from models import VanillaEncoder

device = 'cpu'
model = VanillaEncoder(size=128)

summary(model, (3, 128, 128))

print(np.linspace(0, 1, 10))
