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
from utils import random_sample, standard_vae


def plot_loss(loss_file_name):

    # load file
    loss_file = open(loss_file_name, 'w+')
    # read to numpy array
    loss_array = np.loadtxt(loss_file, delimiter='\t')

    # plot loss
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Average loss')
    plt.plot(loss_array)
    plt.show()


# plots arbitrary number images and their reconstruction
def subplot(images, rec_images, save_as=None):
    """
    :param x: list of arbitrary length with images as np.array
    :param recon_x: list of arbitrary length with reconstructed images
    :param save_as: string, name of plot
    """

    fig, axs = plt.subplots(2, len(images), figsize=(15, 6))
    fig.subplots_adjust(wspace=0, hspace=0)
    axs = axs.ravel()


    axs[0].set_ylabel('Images', rotation=0, size='large')
    axs[5].set_ylabel('Reconstructed Images', rotation=0, size='large')

    for i in range(2*len(images)):
        axs[i].axis('off')
        if i < len(images):
            pic = images[i]
        else:
            pic = rec_images[i-len(images)]
            pic = (pic - np.min(pic))
            pic *= 255/np.amax(pic)
            pic = pic.astype(int)
        axs[i].imshow(pic)

    if save_as is not None:
        plt.savefig('../plots/{}.png'.format(save_as))
    plt.show()

# comment

if __name__ == '__main__':

    # plotting with old VAE
    images, rec_images = random_sample(5)
    subplot(images, rec_images, save_as='testing_old')

    # plotting with Vanillia - delicious
    model_path = '../models/vanilla-4.pth'
    data_dir = '../data/64x64CACD2000'

    model = VanillaVAE(layer_count=3, in_channels=3, latent_dim=100, size=128)

    random_sample(5, model, model_path, data_dir)
    images, rec_images = random_sample(5)
    subplot(images, rec_images, save_as='testing_vanillia')