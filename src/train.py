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

from fcts.load_data import load_data

from tqdm import tqdm_notebook as tqdm



# Training of the VAE
def train(model, epochs, path, optimizer, loss_fct):

    # set pathes to data
    meta_path = '../data/celebrity2000_meta.mat'
    data_dir = '../data/CACD2000'

    # define transformations
    trafo = transforms.Compose([
                        transforms.ToTensor()])

    # datasets
    train_set = FaceDataset(meta_path=meta_path, data_dir=data_dir, transform=trafo)
    test_set = FaceDataset(meta_path=meta_path, data_dir=data_dir, transform=trafo)

    # dataloader
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)

    # check for previous trained models and resume from there if available
    try:
        previous = max(glob.glob(path + '/*.pth'))
        print('load previous model')
        checkpoint = torch.load(previous)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        loss = checkpoint['loss']
        epochs_trained = checkpoint['epoch']
    except Exception as e:
        print('no model to load')
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
            #loss = loss_function(recon_batch,  x, mu, log_var)

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

    # batch size
    batch = 128
    # allow for cuda
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

        # hyperparameters
    encoder_layer_sizes = [32*32, 512, 256]
    decoder_layer_sizes = [256, 512, 32*32]

    latent_dim_baseline = 2
    vae_baseline = VAE_baseline(inp_dim=(32*32), encoder_layer_sizes=encoder_layer_sizes, decoder_layer_sizes=decoder_layer_sizes, latent_dim=latent_dim_baseline)
    vae_baseline = vae_baseline.to(device)
    optimizer_baseline = optim.Adam(vae_baseline.parameters(), lr=1e-3)

    loss_func_baseline = nn.MSELoss()

    epochs_baseline = 15

    train(vae_baseline, epochs_baseline, './models/clustering/baseline', optimizer_baseline, loss_func_baseline)
