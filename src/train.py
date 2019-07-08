import glob
import os.path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from load_data import FaceDataset
from models import VAE
from torch.autograd import Variable
from torch.utils.data import SubsetRandomSampler
from torch.utils.data.dataset import Dataset
from torchvision import models, transforms
from utils import vae_loss, set_split, k_fold_CV
import pandas as pd
from datetime import datetime

# Training of the VAE
def train(model, epochs, batch, optimizer, loss_fct, path, trafo, subset_size=None, test_split=0.2):

    # set pathes to data
    meta_path = '../data/celebrity2000_meta.mat'
    data_dir = '../data/64x64CACD2000'

    # data sets
    dataset = FaceDataset(meta_path=meta_path, data_dir=data_dir, transform=trafo)

    train_indices, test_indices = set_split(len(dataset), subset=subset_size, test_split=test_split)

    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch, sampler=train_sampler)
    test_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch, sampler=test_sampler)


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
        epochs_trained = 0

    model.train()

    for epoch in np.arange(epochs_trained, epochs):

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

        print('====> Epoch: {} Average loss: {:.7f}'.format(epoch, train_loss / len(train_loader.dataset)))

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

        # save model
        dt = datetime.now().replace(microsecond=0)
        torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                }, path+('/{}-{}.pth').format(epoch, dt))


def hyper_search(k, epochs, latent_dim, encoder_params, decoder_params, lr, loss_file_name, trafo, batch=132, subset_size=1000, test_split=0.0):

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


if __name__ == '__main__':

    # needed for macOS in some cases
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    # allow for cuda
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    # hyperparameters
    batch = 132
    epochs = 10
    latent_dim = 50
    ### 64x64 ###
    # here zero padding is needed
    encoder_params = [(3, 32, 4, 2, 1), (32, 64, 4, 2, 1), (64, 128, 4, 2, 1), (128, 256, 4, 2, 1), 256*4*4]
    # here zero padding is needed because kernel of seize three needs padding to retain shape after upsampling
    decoder_params = [256*4*4, (256, 128, 3, 1, 1), (128, 64, 3, 1, 1), (64, 32, 3, 1, 1), (32, 3, 3, 1, 1)]

    ### 32x32 ###
    # here zero padding is needed
    #encoder_params = [(3, 32, 5, 2, 2), (32, 64, 5, 2, 2), (64, 128, 5, 2, 2), 128*5*5]
    # here zero padding is needed because kernel of seize three needs padding to retain shape after upsampling
    #decoder_params = [128*5*5, (128, 64, 5, 2, 2, 1), (64, 32, 5, 2, 2, 1), (32, 3, 5, 2, 2, 1)]

    #lr = 1e-3

    # prepare transformation
    PIL = transforms.ToPILImage()
    normalize = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    to_tensor = transforms.ToTensor()

    # define transformations
    trafo = transforms.Compose([PIL, to_tensor, normalize])

    lr = [.5e-2, .2e-2, 1e-3, .9e-3, .5e-3]

    # set up Model
    #model = VAE(latent_dim, encoder_params, decoder_params)
    #model = model.to(device)
    #optimizer= optim.Adam(model.parameters(), lr=lr)

    #train(model, epochs, batch, optimizer, nn.MSELoss(), './', trafo, subset_size=40000, test_split=0.2)
    hyper_search(3, 5, latent_dim, encoder_params, decoder_params, lr, "./loss/loss_test_30000.csv", trafo, subset_size=1000, test_split=0.2)
