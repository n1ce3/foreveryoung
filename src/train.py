import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.dataset import Dataset
from torchvision import transforms, models
from torch.autograd import Variable

import os.path
import glob
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

from load_data import FaceDataset
from models import VAE
from utils import vae_loss


# Training of the VAE
def train(model, epochs, batch, optimizer, loss_fct):

    # set pathes to data
    meta_path = '../data/celebrity2000_meta.mat'
    data_dir = '../data/CACD2000'

    # not sure if we need normalize, therefore not used in trafo
    PIL = transforms.ToPILImage()
    normalize = transforms.Normalize(mean=0, std=1)
    grey = transforms.Grayscale(num_output_channels=1)
    crop = transforms.CenterCrop(size=64)
    to_tensor = transforms.ToTensor()

    # define transformations
    trafo = transforms.Compose([PIL, grey, crop, to_tensor])

    # datasets
    train_set = FaceDataset(meta_path=meta_path, data_dir=data_dir, transform=trafo)
    test_set = FaceDataset(meta_path=meta_path, data_dir=data_dir, transform=trafo)

    # dataloader
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch, shuffle=False)

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
            x = data
            x = x.to(device)
            optimizer.zero_grad()

            recon_batch,  mu, log_var = model(x)
            loss = vae_loss(recon_batch,  x, mu, log_var, loss_fct)

            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))

        # save model
        torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                }, path+('/vae-{}.pth').format(epoch))

if __name__ == '__main__':

    # needed for macOS in some cases
    os.environ['KMP_DUPLICATE_LIB_OK']='True'

    # allow for cuda
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    # hyperparameters
    batch = 8
    epochs = 10
    latent_dim = 10
    encoder_params = [(1, 32, 4, 2), (32, 64, 4, 2), (64, 128, 4, 2), (128, 256, 4, 2), 256*4*4]
    # here zero padding is needed because kernel of seize three needs padding to retain shape after upsampling
    decoder_params = [256*4*4, (256, 128, 3, 1, 1), (128, 64, 3, 1, 1), (64, 32, 3, 1, 1), (32, 1, 3, 1)]
    lr = 1e-4

    # set up Model
    model = VAE(latent_dim, encoder_params, decoder_params)
    model = model.to(device)
    optimizer= optim.Adam(model.parameters(), lr=lr)

    train(model, epochs, batch, optimizer, nn.MSELoss())
