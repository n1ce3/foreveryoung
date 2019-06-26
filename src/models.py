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
        super(CVAE, self).__init__()

        self.latent_dim = latent_dim

        self.encoder = Encoder(encoder_layer_sizes, latent_dim)
        self.decoder = Decoder(decoder_layer_sizes, latent_dim)

    def forward(self, x):

    def sampling():


class Encoder():

    def __init__():

    def forward():

class Decoder():

    def __init__():

    def forward():

    