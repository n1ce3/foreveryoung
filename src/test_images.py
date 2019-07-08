import numpy as np
from numpy.random import normal
import glob
import os.path
from operator import itemgetter
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import SubsetRandomSampler
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset
from torchvision import models, transforms

from load_data import FaceDataset
from models import VAE
from utils import plot_instances, standard_vae

if __name__ == "__main__":

    # needed for macOS in some cases
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    # allow for cuda
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    # define transformations
    PIL = transforms.ToPILImage()
    normalize = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    to_tensor = transforms.ToTensor()
    trafo = transforms.Compose([PIL, to_tensor, normalize])

    model_path = '../models/9-2019-07-08 17:47:48.pth'
    meta_path = '../data/celebrity2000_meta.mat'
    data_dir = '../data/64x64CACD2000'

    plot_instances(10, standard_vae(), model_path, meta_path, data_dir, trafo)
