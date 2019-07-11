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
from models import VanillaVAE
from utils import random_sample, random_interpolate, age_interpolate, sample_instances


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



# plots arbitrary number images and their reconstruction
def plot_interpolations(interpolations, save_as=None):
    """
    :param interpolations: list of arbitrary length with images as np.array
    :param save_as: string, name of plot
    """

    fig, axs = plt.subplots(1, len(interpolations), figsize=(15, 6))
    fig.subplots_adjust(wspace=0, hspace=0)
    axs = axs.ravel()

    for i in range(len(interpolations)):
        axs[i].axis('off')
        pic = interpolations[i]
        pic = (pic - np.min(pic))
        pic *= 255/np.amax(pic)
        pic = pic.astype(int)
        axs[i].imshow(pic)

    if save_as is not None:
        plt.savefig('../plots/{}.png'.format(save_as))
    plt.show()

def plot_samples(samples, save_as):

    fig, axs = plt.subplots(len(samples), len(samples[0]), figsize=(15, 3.2))
    fig.subplots_adjust(wspace=0, hspace=0)
    axs = axs.ravel()
    

    samples_flat = [item for sublist in samples for item in sublist]

    for i in range(len(samples)*len(samples[0])):
        axs[i].axis('off')
        pic = samples_flat[i]
        pic = (pic - np.min(pic))
        pic *= 255/np.amax(pic)
        pic = pic.astype(int)
        axs[i].imshow(pic)

    if save_as is not None:
        plt.savefig('../plots/{}.png'.format(save_as),bbox_inches='tight', transparent="True", pad_inches=0)
    plt.show()



if __name__ == '__main__':

    # plotting with old VAE
    #images, rec_images = random_sample(5)
    #subplot(images, rec_images, save_as='testing_old')

    # plotting with Vanillia - delicious
    model_path = '../models/Vanilla_128_lr5e-4stable_cont_lr1-4-19.pth'
    data_dir = '../data/128x128CACD2000'

    model = VanillaVAE(layer_count=4, in_channels=3, latent_dim=100, size=128)

    #images, rec_images = random_sample(5, model, model_path=model_path, data_dir=data_dir)
    #subplot(images, rec_images, save_as='testing_vanillia_layer4_size64_stableLR_20epochs_final')

    # here interpolation is done
    interpolations1 = random_interpolate(5, model, model_path, subset=None, test_split=0.2)
    interpolations2 = random_interpolate(5, model, model_path, subset=None, test_split=0.2)

    # here interpolation between age is done
    file1 = '14_Emma_Watson_0008.jpg'
    file2 = '23_Emma_Watson_0016.jpg'
    file3 = '46_Michelle_Pfeiffer_0008.jpg'
    file4 = '55_Michelle_Pfeiffer_0016.jpg'
    interpolations_age1 = age_interpolate(5, model, model_path, file1, file2)
    interpolations_age2 = age_interpolate(5, model, model_path, file3, file4)

    # here generation is done
    #samples1 = sample_instances(5, model, model_path)
    #samples2 = sample_instances(5, model, model_path)

    #plot_interpolations(interpolations, save_as='random_interpolation_final')

    #plot_interpolations(interpolations_age, save_as='first_interpolation')
    plot_samples([interpolations_age1, interpolations_age2], save_as='random_interpolation_age_final_fem')
