import numpy as np
from numpy.random import normal
import glob
import os.path
import os
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
from models import VAE, VanillaVAE

# Implement function to plot random images and their reconstruction
def plot_instances(n, model, model_path, meta_path, data_dir, transform, test_split=0.2, subset=None):

    # restore model
    weights = torch.load(model_path)
    model.load_state_dict(weights['model_state_dict'])

    # load data and sample n random images
    filelist = glob.glob(data_dir+'/*.jpg')
    size = subset if subset is not None else len(filelist)
    _, test_sampler = set_split(size, test_split=test_split)

    test_sampler = np.array(test_sampler)
    filelist = np.array(filelist)

    # undo seed
    np.random.seed(seed=None)
    # shuffel
    np.random.shuffle(test_sampler)

    # choose right files
    sub_filelist = filelist[test_sampler[:n]]

    samples = []
    # iterate files and append to list
    for i, filename in enumerate(sub_filelist):
        im = Image.open(filename)
        samples.append(np.array(im))

    rec_samples = []
    # reconstruct images using VAE
    for i, pic in enumerate(samples):
        reconstructed = model(transform(np.array(pic)).unsqueeze(0))[0][0]
        rec_samples.append(reconstructed.permute(1, 2, 0).detach().numpy())

    plt.figure()
    # plot images
    for i, pic in enumerate(rec_samples):
        print('pic: ', i)
        pic = (pic - np.min(pic))
        pic *= 255/np.amax(pic)
        pic = pic.astype(int)
        plt.imshow(pic)
        plt.show()

# plots arbitrary number of np.arrays in array x
def subplot(x, recon_x, save_as=None):
    """
    :param x: list of arbitrary length with images as np.array
    :param recon_x: list of arbitrary length with reconstructed images
    :param save_as: string, name of plot
    """

    fig, axs = plt.subplots(2, len(x), figsize=(15, 6))
    fig.subplots_adjust(wspace=0, hspace=0)
    axs = axs.ravel()


    axs[0].set_ylabel('Images', rotation=0, size='large')
    axs[5].set_ylabel('Reconstructed Images' rotation=0, size='large')

    for i in range(2*len(x)):
        axs[i].axis('off')
        axs[i].imshow(x[i]) if i < len(x) else axs[i].imshow(recon_x[i])

    plt.show()
    if save_as is not None:
        plt.savefig('../plots/{}.png'.format(save_as))

if __name__ == '__main__':
    return 0
