import torch
import numpy as np
from models import VAE
from torch.utils.data import SubsetRandomSampler
from torchvision import models, transforms
import glob
from operator import itemgetter
from PIL import Image

from tqdm import tqdm
import matplotlib.pyplot as plt


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
def plot_instances(n, model, model_path, meta_path, data_dir, transform):

    # restore model
    weights = torch.load(model_path)
    model.load_state_dict(weights['model_state_dict'])

    # load data and sample n random images
    filelist = glob.glob(data_dir+'/*.jpg')
    _, test_sampler = set_split(40000)
    # choose right files
    sub_filelist = itemgetter(*list(np.random.choice(test_sampler, n)))(filelist)

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
        print(np.amin(pic))
        print(np.amax(pic))
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



if __name__ == "__main__":
    # not sure if we need normalize, therefore not used in trafo
    PIL = transforms.ToPILImage()
    normalize = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    #grey = transforms.Grayscale(num_output_channels=1)
    #crop = transforms.CenterCrop(size=32)
    to_tensor = transforms.ToTensor()

    # define transformations
    trafo = transforms.Compose([PIL, to_tensor, normalize])

    # plot
    model_path = './vae-9.pth'
    meta_path = '../data/celebrity2000_meta.mat'
    data_dir = '../data/64x64CACD2000'
    latent_dim = 100
    ### 64x64 ###
    # here zero padding is needed
    encoder_params = [(3, 32, 4, 2, 1), (32, 64, 4, 2, 1), (64, 128, 4, 2, 1), (128, 256, 4, 2, 1), 256*4*4]
    # here zero padding is needed because kernel of seize three needs padding to retain shape after upsampling
    decoder_params = [256*4*4, (256, 128, 3, 1, 1), (128, 64, 3, 1, 1), (64, 32, 3, 1, 1), (32, 3, 3, 1, 1)]

    # set up Model
    model = VAE(latent_dim, encoder_params, decoder_params)

    plot_instances(10, model, model_path, meta_path, data_dir, trafo)