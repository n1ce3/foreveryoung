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
        pass

    def sampling():
        pass

class Encoder(nn.Module):

    def __init__(self, layer_sizes, latent_dim):
        """
        Arguments:
            layer_sizes (list[tupel(int)]): list containing tupels of sizes of the convolutional encoder layers,
            latent_dim (int): latent space dimension
        """
        super(Encoder, self).__init__()

        self.latent_dim = latent_dim

        # initialize layers
        layer_list = []
        for shape in layer_sizes:
            layer_list.append(nn.Conv2d(*shape))
        # store layers
        self.layers = nn.Sequential(*layer_list)

        # layers for latent space output
        last_conv = layer_sizes[-1]
        self.out_mean = nn.Linear(last_conv[0]*last_conv[0]*last_conv[1], latent_dim)
        self.out_var = nn.Linear(last_conv[0]*last_conv[0]*last_conv[1], latent_dim)


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

    def __init__(self, layer_sizes, latent_dim):
        """
        """
        super(Decoder, self).__init__()

        self.latent_dim = latent_dim


    def forward():
        pass

    