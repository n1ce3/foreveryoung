import torch
import numpy as np
from torch.utils.data import SubsetRandomSampler


# Implement the loss function for the VAE
def vae_loss(recon_x, x, mu, log_var, loss_func):
    """
    :param recon_x: reconstruced input
    :param x: input
    :param mu, log_var: parameters of posterior (distribution of z given x)
    :loss_func: loss function to compare input image and constructed image
    """
    recon_loss = loss_func(recon_x, x)
    kl_loss = torch.mean(0.5 * torch.sum(
        torch.exp(log_var) + mu**2 - 1. - log_var, 1))
    return recon_loss + kl_loss

 # function to split data into train and test set
def set_split(data_set, subset=None, test_split=0.2, SEED = 42):
    # size of data set
    size = len(data_set)
    # all indices
    indices = list(range(size))
    # split
    split = int(np.floor(test_split*size)) if subset is None else int(np.floor(test_split*subset))
    # shuffle indices
    np.random.shuffle(indices)
    np.random.seed(SEED)

    train_indices, test_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    return train_sampler, test_sampler


# Implement function to plot random images and their reconstruction
#def plot_instances(n):
    # load data and sample n random images

    # reconstruct images using VAE

    # plot images
    