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

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

model = VanillaVAE(layer_count=4, in_channels=3, latent_dim=100, size=128, name='trying_shit')
model.to(device)

summary(model, (3, 128, 128))
