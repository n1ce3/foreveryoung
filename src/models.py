import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# variational autoencoder
class VAE(nn.Module):

    def __init__(self, latent_dim, encoder_layer_sizes, decoder_layer_sizes):
        """
        Arguments:
            encoder_layer_sizes (list[tupel(int)]): list containing tupels of sizes of the encoder layers,
            decoder_layer_sizes (list[tupel(int)]): list containing tupels of sizes of the decoder layers,
            latent_dim (int): latent space dimension
        """
        super(VAE, self).__init__()

        self.latent_dim = latent_dim

        self.encoder = Encoder(encoder_layer_sizes, latent_dim)
        self.decoder = Decoder(decoder_layer_sizes, latent_dim)

    def forward(self, x):
        """
        Forward Process of whole VAE. 
        Arguments:
            x: tensor of dimension (batch_size, 1, input_shape, input_shape)
        Output: recon_x, means, log_var
            recon_x: reconstruction of input,
            means: output of encoder,
            log_var: output of encoder
        """
        
        means, log_vars = self.encoder(x)
        std = torch.exp(.5*log_vars)
        eps = torch.randn_like(std)
        z = means + eps*std
        recon_x = self.decoder(z)

        return recon_x, means, log_vars

    def sampling(self, n=2):
        """
        Arguments:
            n (int): amount of samples (amount of elements in the latent space)
        Output:
            x_sampled: n randomly sampled elements of the output distribution
        """
        # draw samples p(z)~N(0,1)
        z = torch.randn((n, self.latent_dim))
        # generate
        x_sampled = self.decoder(z, c)

        return x_sampled


class Encoder(nn.Module):

    def __init__(self, kernel_sizes, latent_dim):
        """
        Arguments:
            kernel_sizes (list[tupel(int)]): list containing tupels of sizes of the convolutional encoder kernels
                                             last element is size of fully connected layer
            latent_dim (int): latent space dimension
        """
        super(Encoder, self).__init__()

        self.latent_dim = latent_dim

        # initialize convolutional layers
        layer_list = []
        for shape in kernel_sizes[:-1]:
            layer_list.append(nn.Conv2d(*shape))
        # store layers
        self.layers = nn.Sequential(*layer_list)

        # layers for latent space output
        self.out_mean = nn.Linear(kernel_sizes[-1], latent_dim)
        self.out_var = nn.Linear(kernel_sizes[-1], latent_dim)

    def forward(self, x):
        # forward 
        out = self.layers(x)

        # reshape
        out = out.view(out.size()[0], -1)

        # latent space output
        means = self.out_mean(out)
        log_vars = self.out_var(out)

        return means, log_vars


class Decoder(nn.Module):

    def __init__(self, kernel_sizes, latent_dim):
        """
        Arguments:
            kernel_sizes (list[int]): first entry is output size of fully connected layer
                                      all others are shape of upsampling convolutional layers,
            latent_dim (int): dimension of latent space, i.e. dimension out input of the decoder,
        """
        super(Decoder, self).__init__()

        self.latent_dim = latent_dim

        # initialize fully connected layers
        layer_list.append(nn.Linear(latent_dim, kernel_sizes[0]))

        # initialize convolutional layers
        layer_list = []
        for shape in kernel_sizes[1:]:
            layer_list.append(nn.ConvTranspose2d(*shape))

        # store layers
        self.layers = nn.Sequential(*layer_list)

    def forward(self, z):
        return self.layers(z)

    