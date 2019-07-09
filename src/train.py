import numpy as np
import glob
import sys
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
from models import VAE, VanillaVAE
from utils import vae_loss, set_split, k_fold_CV, hyper_search, standard_vae, vae_loss_MSE

# Training of the VAE
def train(model, epochs, batch, trafo, subset_size=None, test_split=0.2, load=False, lrs=[0.001], alpha=0.1):

    # set pathes to data
    meta_path = '../data/celebrity2000_meta.mat'
    data_dir = '../data/64x64CACD2000'

    # data sets
    dataset = FaceDataset(meta_path=meta_path, data_dir=data_dir, transform=trafo, subset=subset_size)

    train_indices, test_indices = set_split(len(dataset), test_split=test_split)

    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch, sampler=train_sampler)
    test_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch, sampler=test_sampler)

    epochs_trained = 0
    if load:
        # check for previous trained models and resume from there if available
        try:
            previous = max(glob.glob('../models/*.pth'))
            print('Load previous model')
            checkpoint = torch.load(previous)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            loss = checkpoint['loss']
            epochs_trained = checkpoint['epoch']
        except Exception as e:
            print('No model to load')

    model.train()

    for epoch in np.arange(epochs_trained, epochs):

        # set up optimizer with array of learning rates
        if epoch >= len(lrs):
            lr = lrs[-1]
        else:
            lr = lrs[epoch]
        print('Learning rate is: {}'.format(lr))

        optimizer = optim.Adam(model.parameters(), lr=lr)

        train_loss = 0

        for batch_idx, data in enumerate(tqdm(train_loader, desc=f'Train Epoch {epoch}', leave=False)):
            # we only need the image for the moment
            x = data['image']
            x = x.to(device)
            optimizer.zero_grad()

            recon_batch,  mu, log_var = model(x)
            loss = vae_loss_MSE(recon_batch,  x, mu, log_var, alpha=alpha)

            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        print('====> Epoch: {} Average loss: {:.7f}'.format(epoch, train_loss / len(train_loader.dataset)))

        # test model after each epoch to track progress
        test(model, test_loader)

        # save model
        dt = datetime.now().replace(microsecond=0)
        torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                }, ('../models/{}-{}-{}.pth').format(model.name, dt, epoch))

def test(model, test_loader):

    model.eval()
    test_loss = 0
    for batch_idx, data in enumerate(tqdm(test_loader, desc='Test', leave=False)):
        # reconstruct image x
        x = data['image']
        x = x.to(device)
        recon_batch,  mu, log_var = model(x)
        # get loss
        loss = vae_loss_MSE(recon_batch,  x, mu, log_var, alpha=alpha)
        test_loss += loss.item()

    # always return stuff how u got it :)
    model.train()
    print('====> Average Test loss: {:.7f}'.format(test_loss / len(test_loader.dataset)))


if __name__ == '__main__':

    # needed for macOS in some cases
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    # allow for cuda
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    # hyperparameters
    batch = 128
    epochs = 10

    # option to pass lrs array as argument to main
    if len(sys.argv) >= 2:
        lrs = []
        for lr in sys.argv[1:]:
            lrs.append(float(lr))
        print('Lrs passed: {}'.format(lrs))

    else:
        lrs = np.logspace(-3, -4, 10)

    lrs_search = [.5e-2, .2e-2, 1e-3, .9e-3, .5e-3]
    # alpha determines the contribution of KL-Loss, alpha = 0 means no KL-Loss
    alpha = 0.1

    ### 32x32 ###
    #encoder_params = [(3, 32, 5, 2, 2), (32, 64, 5, 2, 2), (64, 128, 5, 2, 2), 128*5*5]
    #decoder_params = [128*5*5, (128, 64, 5, 2, 2, 1), (64, 32, 5, 2, 2, 1), (32, 3, 5, 2, 2, 1)]

    # prepare transformation
    PIL = transforms.ToPILImage()
    normalize = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    to_tensor = transforms.ToTensor()

    # define transformations
    trafo = transforms.Compose([PIL, to_tensor, normalize])

    # set up Model
    #model = standard_vae()
    model = VanillaVAE(layer_count=3, in_channels=3, latent_dim=100, size=128)

    train(model, epochs, batch, trafo, test_split=0.2, lrs=lrs, alpha=alpha)

    # Hyperparameter search

    # hyper_search(3, 5, latent_dim, encoder_params, decoder_params, lrs_search, "./loss/loss_test_30000.csv", trafo, subset_size=1000, test_split=0.2)
