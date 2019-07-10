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


# Implement the loss function for the VAE
def vae_loss(recon_x, x, mu, log_var, loss_func, alpha=1.0):
    """
    :param recon_x: reconstruced input
    :param x: input
    :param mu, log_var: parameters of posterior (distribution of z given x)
    :loss_func: loss function to compare input image and constructed image
    """
    recon_loss = loss_func(recon_x, x)
    kl_loss = torch.mean(0.5 * torch.sum(
        torch.exp(log_var) + mu**2 - 1. - log_var, 1))
    #print('Pixel Loss: {}, KL-Loss: {}'.format(recon_loss, kl_loss))
    return alpha*recon_loss + kl_loss

def vae_loss_MSE(recon_x, x, mu, log_var, alpha=1.0):
    recon_loss = torch.mean((recon_x - x)**2)
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_loss = -0.5 * torch.mean(torch.mean(1 + log_var - mu.pow(2) - log_var.exp(), 1))
    #print('Pixel Loss: {}, KL-Loss: {}'.format(recon_loss, kl_loss))
    return recon_loss + kl_loss*alpha

 # function to split data into train and test set
def set_split(size_dataset, test_split=0.2, SEED=42):
    """
    :param size_dataset (int): size of dataset
    :param split (tupel): tupel of split in (trains_size, val_size, test_size), sum of tupel need to be 1
    :param SEED (int): Seed for random operations
    """
    # all indices
    indices = list(range(size_dataset))
    # shuffle indices
    np.random.seed(SEED)
    np.random.shuffle(indices)

    test_split = np.floor(size_dataset*test_split).astype(int)

    # get indices for split
    test_indices, train_indices = indices[:test_split], indices[test_split:]

    return train_indices, test_indices

def k_fold_CV(data_indices, k=5):

    size = len(data_indices)

    # split data in k-folds
    folds = np.array_split(np.array(data_indices), k)

    train_indices = []
    val_indices = []
    for k_ in range(k):
        # copy folds
        folds_ = folds.copy()
        # val set
        val_indices.append(folds[k_])
        # remove from folds_
        del folds_[k_]
        # train set
        train_indices.append([item for sublist in folds_ for item in sublist])

    return train_indices, val_indices


# Implement function to plot random images and their reconstruction
def plot_instances(n, model, model_path, meta_path, data_dir, transform, test_split=0.2, subset=None):

    # restore model
    weights = torch.load(model_path, map_location='cpu')
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

# returns a vae with standard parameters, convenience functions to keep code clean
def standard_vae():
    # allow for cuda
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    ### 64x64 ###
    encoder_params = [(3, 32, 4, 2, 1), (32, 64, 4, 2, 1), (64, 128, 4, 2, 1), (128, 256, 4, 2, 1), 256*4*4]
    decoder_params = [256*4*4, (256, 128, 3, 1, 1), (128, 64, 3, 1, 1), (64, 32, 3, 1, 1), (32, 3, 3, 1, 1)]
    latent_dim = 100

    # set up Model
    model = VAE(latent_dim, encoder_params, decoder_params)
    model = model.to(device)
    return model

def standard_vae32():

    # allow for cuda
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    ### 32x32 ###
    encoder_params = [(3, 32, 5, 2, 2), (32, 64, 5, 2, 2), (64, 128, 5, 2, 2), 128*5*5]
    decoder_params = [128*5*5, (128, 64, 5, 2, 2, 1), (64, 32, 5, 2, 2, 1), (32, 3, 5, 2, 2, 1)]
    latent_dim = 100

    # set up Model
    model = VAE(latent_dim, encoder_params, decoder_params)
    model = model.to(device)
    return model

def hyper_search(k, epochs, latent_dim, encoder_params, decoder_params, lrs, loss_file_name, trafo, batch=132, subset_size=1000, test_split=0.0):

    # build dataframe
    df = pd.DataFrame.from_dict({})

    # prepare data
    # set paths to data
    meta_path = '../data/celebrity2000_meta.mat'
    data_dir = '../data/64x64CACD2000'

    # data sets
    dataset = FaceDataset(meta_path=meta_path, data_dir=data_dir, transform=trafo, subset=subset_size)

    # get data subset
    train_indices, _ = set_split(len(dataset), test_split=test_split)

    # save loss

    for lr in tqdm(lr, desc='Hyperparameter search', leave=False):

        print('Learning rate: ', lr)

        # CV split
        cv_train, cv_val = k_fold_CV(train_indices, k=k)

        # loss
        temp_loss = np.zeros((k, epochs))

        for k_ in range(len(cv_train)):
            print('Fold {} of {}'.format(k_+1, k))
            # load data for split
            train_sampler = SubsetRandomSampler(cv_train[k_])
            test_sampler = SubsetRandomSampler(cv_val[k_])
            train_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch, sampler=train_sampler)
            test_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch, sampler=test_sampler)

            # initialize model
            model = VAE(latent_dim, encoder_params, decoder_params)
            model = model.to(device)
            optimizer = optim.Adam(model.parameters(), lr=lr)
            loss_fct = nn.MSELoss()

            model.train()

            # train model
            for epoch in range(epochs):
                train_loss = 0
                for batch_idx, data in enumerate(tqdm(train_loader, desc=f'Train Epoch {epoch}', leave=False)):
                    # we only need the image for the moment
                    x = data['image']
                    x = x.to(device)
                    optimizer.zero_grad()
                    recon_batch,  mu, log_var = model(x)
                    loss = vae_loss(recon_batch,  x, mu, log_var, loss_fct)
                    loss.backward()
                    train_loss += loss.item()
                    optimizer.step()

                # test model after each epoch to track progress
                test_loss = 0
                for batch_idx, data in enumerate(tqdm(test_loader, desc='Test', leave=False)):
                    # reconstruct image x
                    x = data['image']
                    x = x.to(device)
                    recon_batch,  mu, log_var = model(x)
                    # get loss
                    loss = vae_loss(recon_batch,  x, mu, log_var, loss_fct)
                    test_loss += loss.item()
                print('====> Average Test loss: {:.7f}'.format(test_loss / len(test_loader.dataset)))
                # save loss
                temp_loss[k_, epoch] = test_loss / len(test_loader.dataset)

            # add to df
            df[str(lr)] = np.mean(temp_loss, axis=0)
            # save temp_loss to file
            df.to_csv(loss_file_name, sep='\t')


'Get newest file in folder'
def newest(path='../models/'):
    files = os.listdir(path)
    paths = [os.path.join(path, basename) for basename in files]
    return max(paths, key=os.path.getctime)


def random_sample(n, model=standard_vae(), model_path=newest(), data_dir='../data/64x64CACD2000', subset=10000, test_split=0.2):
    """
    :param x: list of arbitrary length with images as np.array
    :param recon_x: list of arbitrary length with reconstructed images
    :param save_as: string, name of plot
    returns list of images and list of reconstructed images
    """

    # restore model
    weights = torch.load(model_path)
    model.load_state_dict(weights['model_state_dict'])

    meta_path = '../data/celebrity2000_meta.mat'

    PIL = transforms.ToPILImage()
    normalize = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    to_tensor = transforms.ToTensor()
    transform = transforms.Compose([PIL, to_tensor, normalize])

    # load data and sample n random images
    filelist = glob.glob(data_dir+'/*.jpg')
    size = subset if subset is not None else len(filelist)
    _, test_sampler = set_split(size, test_split=test_split)

    # plot
    model_path = '../models/vanilla-9.pth'
    meta_path = '../data/celebrity2000_meta.mat'
    data_dir = '../data/128x128CACD2000'

    model = VanillaVAE(layer_count=4, in_channels=3, latent_dim=100, size=128)

    return images, rec_images
