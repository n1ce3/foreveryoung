import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

# variational autoencoder
class VAE(nn.Module):

    def __init__(self, latent_dim, encoder_layer_sizes, decoder_layer_sizes, name='standard'):
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
        self.name = name

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

        means, log_vars = self.encoder(x.view(-1, 3, 64, 64))
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
        x_sampled = self.decoder(z)

        return x_sampled

    def interpolate(self, x1, x2, n):
        """
        Interpolating between two images using VAE
        Arguments:
            x1: tensor of dimension (1, 3, input_shape, input_shape)
            x2: tensor of dimension (3, input_shape, input_shape)
            n: number of interpolation between x1 and x2
        Output: interpolations
            interpolations: list of interpolations between x1 and x2, first
            value is x1 and last value is x2
        """
        means1, log_vars1 = self.encoder(x1)
        means2, log_vars2 = self.encoder(x2)

        interpolations = []

        # interpolating with parameter alpha
        for alpha in np.linspace(0, 1, n+2):
            means = (1-alpha)*means1 + alpha*means2
            log_vars = (1-alpha)*log_vars1 + alpha*log_vars2

            std = torch.exp(.5*log_vars)
            eps = torch.randn_like(std)
            z = means + eps*std
            interpolations.append(self.decoder(z)).unsqueeze(0))[0][0]

        return interpolations


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
            # batchnorm requires number of features
            layer_list.append(nn.BatchNorm2d(shape[1]))
            # layer_list.append(nn.ReLU())
            layer_list.append(nn.LeakyReLU())
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
                                      all others are shape of upsampling convolutional kernels,
            latent_dim (int): dimension of latent space, i.e. dimension out input of the decoder,
        """
        super(Decoder, self).__init__()

        self.latent_dim = latent_dim
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        # initialize convolutional layers
        layer_list = []

        # initialize fully connected layers
        self.linear = nn.Linear(latent_dim, kernel_sizes[0])

        for shape in kernel_sizes[1:]:
            # layer_list.append(nn.ReLU())
            layer_list.append(nn.LeakyReLU())
            layer_list.append(nn.Upsample(scale_factor=2))
            layer_list.append(nn.Conv2d(*shape))

        # store layers
        self.layers = nn.Sequential(*layer_list)

    def forward(self, z):
        out = self.linear(z)
        # reshape
        out = out.view(out.size()[0], -1, 4, 4)
        #out = nn.Tanh()(self.layers(out))
        out = self.layers(out)
        return out



##### NEW VAE #####
class VanillaVAE(nn.Module):

    def __init__(self, layer_count=3, in_channels=3, latent_dim=100, size=64, name='vanilla'):
        super(VanillaVAE, self).__init__()

        self.size = size
        self.layer_count = layer_count
        self.latent_dim = latent_dim
        self.in_channels = in_channels

        self.name = name

        self.encoder = VanillaEncoder(layer_count, in_channels, latent_dim, size)
        self.decoder = VanillaDecoder(layer_count, in_channels, size, self.encoder.size_max, latent_dim)

    def forward(self, x):
        means, log_vars = self.encoder(x)
        means = means.squeeze()
        log_vars = log_vars.squeeze()

        std = torch.exp(.5*log_vars)
        eps = torch.randn_like(std)
        z = eps.mul(std).add_(means)

        recon_x = self.decoder(z.view(-1, self.latent_dim, 1, 1))

        return recon_x, means, log_vars

    def interpolate(self, x1, x2, n):
        """
        Interpolating between two images using VAE
        Arguments:
            x1: tensor of dimension (1, 3, input_shape, input_shape)
            x2: tensor of dimension (3, input_shape, input_shape)
            n: number of interpolation between x1 and x2
        Output: interpolations
            interpolations: list of interpolations between x1 and x2, first
            value is x1 and last value is x2
        """
        means1, log_vars1 = self.encoder(x1)
        means2, log_vars2 = self.encoder(x2)

        interpolations = []

        # interpolating with parameter alpha
        for alpha in np.linspace(0, 1, n+2):
            means = (1-alpha)*means1 + alpha*means2
            log_vars = (1-alpha)*log_vars1 + alpha*log_vars2

            std = torch.exp(.5*log_vars)
            eps = torch.randn_like(std)
            z = means + eps*std
            interpolations.append(self.decoder(z)).unsqueeze(0))[0][0]

        return interpolations

class VanillaEncoder(nn.Module):

    def __init__(self, layer_count=3, in_channels=3, latent_dim=100, size=64):
        super(VanillaEncoder, self).__init__()

        self.size = size
        self.layer_count = layer_count
        self.latent_dim = latent_dim
        self.in_channels = in_channels

        # initialize convolutional layers
        layer_list = []

        # append first layers
        mul = 1
        inputs = in_channels

        for i in range(layer_count):
            layer_list.append(nn.Conv2d(inputs, size*mul, 4, 2, 1))
            layer_list.append(nn.BatchNorm2d(size*mul))
            layer_list.append(nn.ReLU())
            inputs = size*mul
            mul *= 2

        self.size_max = inputs

        # append linear layer
        self.out_mean = nn.Linear(inputs*4*4, latent_dim)
        self.out_var = nn.Linear(inputs*4*4, latent_dim)

        # store layers
        self.layers = nn.Sequential(*layer_list)

    def forward(self, x):
        # forward
        out = self.layers(x)
        # reshape
        out = out.view(-1, self.size_max*4*4)
        # latent space output
        means = self.out_mean(out)
        log_vars = self.out_var(out)

        return means, log_vars


class VanillaDecoder(nn.Module):

    def __init__(self, layer_count, in_channels, size, max_size, latent_dim):
        super(VanillaDecoder, self).__init__()

        self.size = size
        self.max_size = max_size
        self.latent_dim = latent_dim

        self.leaky_relu = nn.LeakyReLU(0.2)

        self.linear = nn.Linear(latent_dim, max_size*4*4)

        layer_list = []

        inputs = max_size

        mul = inputs // size // 2
        for i in range(1, layer_count):
            layer_list.append(nn.ConvTranspose2d(inputs, size*mul, 4, 2, 1))
            layer_list.append(nn.BatchNorm2d(size*mul))
            layer_list.append(nn.LeakyReLU(0.2))
            inputs = size*mul
            mul //= 2

        layer_list.append(nn.ConvTranspose2d(inputs, in_channels, 4, 2, 1))

        # store layers
        self.layers = nn.Sequential(*layer_list)

    def forward(self, z):
        z = z.view(z.shape[0], self.latent_dim)
        out = self.linear(z)
        out = out.view(-1, self.max_size, 8, 8)
        out = self.leaky_relu(out)
        out = self.layers(out)
        out = nn.Tanh()(out)
        return out
