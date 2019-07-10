from datetime import datetime
from numpy.random import normal
import numpy as np
import torch
from models import VAE, VanillaVAE
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

file_names =

matt_damon_young = '34_Matt_Damon_0001.jpg'
matt_damon_old1 = '42_Matt_Damon_0010.jpg'
matt_damon_old2 = '42_Matt_Damon_0011.jpg'

daniel_radcliff_young = '15_Daniel_Radcliffe_0001.jpg'
daniel_radcliff_old1 = '24_Daniel_Radcliffe_0017.jpg'
daniel_radcliff_old2 = '24_Daniel_Radcliffe_0005.jpg'
