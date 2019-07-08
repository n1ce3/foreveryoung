import numpy as np
import scipy.io as sio
from PIL import Image
import glob
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import os
from operator import itemgetter


# used for exploration and understanding of the structure of the meta data
def explore_meta(meta_path, print_info=True):
    # read the file into numpy
    meta = sio.loadmat(meta_path)

    if print_info:
        print(type(meta))
        print(meta.keys())
        for key in meta.keys():
            print(key, type(meta[key]))

    # Keys which store data
    key1 = 'celebrityData'
    key2 = 'celebrityImageData'

    if print_info:
        # Shape of numpy arrays
        print(key1, np.shape(meta[key1]))
        print(key2, np.shape(meta[key2]))

        # Data seems to be in the [0, 0] entry
        print(key1, np.shape(meta[key1][0, 0]))
        print(key2, np.shape(meta[key2][0, 0]))

    cData = meta[key1][0, 0]
    cImageData = meta[key2][0, 0]

    # cData columns as they are structured
    name = cData[0]
    cel_id = cData[1]
    birth = cData[2]
    rank = cData[3]
    in_lfw = cData[4]

    age = cImageData[0]
    im_celeb_id = cImageData[1]
    year = cImageData[2]
    # column 3 contains feature - empty in dataset metadata only
    # column 4 contains rank
    # column 5 contains in_lfw
    # column 6 contains birth year
    file_name = cImageData[7]

    print(file_name[5][0][0], type(file_name[5][0][0]))
    print(name[5][0][0])


# returns the name of a celebrity given the corresponding id
def return_name(idx, meta_path):
    meta = sio.loadmat(meta_path)
    key1 = 'celebrityData'
    cData = meta[key1][0, 0]
    # idx must be lowered by one because they are not 0 based
    if idx > 2000:
        raise ValueError('Idx must not exceed 2000')
    return cData[0][idx-1][0][0][0]


# used to test the basic function of the dataset
def test_dataset(meta_path, data_dir):
    face_dataset = FaceDataset(meta_path=meta_path, data_dir=data_dir)

    fig = plt.figure()

    for i in range(len(face_dataset)):
        sample = face_dataset[i+100]

        print(sample['name'], sample['image'].shape, sample['age'])

        ax = plt.subplot(1, 4, i + 1)
        plt.tight_layout()
        ax.set_title('Sample #{}'.format(i))
        ax.axis('off')
        ax.imshow(sample['image'])

        if i == 3:
            plt.show()
            break


class FaceDataset(Dataset):
    """Face dataset."""

    def load_meta(self, meta_path):
        # load meta data
        meta = sio.loadmat(meta_path)

        # Keys which store data
        key1 = 'celebrityData'
        key2 = 'celebrityImageData'

        # access meta data
        cData = meta[key1][0, 0]
        cImageData = meta[key2][0, 0]

        # relevant columns of image data, cData not needed in Dataset
        age = cImageData[0]
        celeb_id = cImageData[1]
        year = cImageData[2]
        file_name = cImageData[7]

        if self.subset is not None:
            age = age[self.indices]
            celeb_id = celeb_id[self.indices]
            year = year[self.indices]
            file_name = file_name[self.indices]

        return {'age': age, 'celeb_id': celeb_id, 'year': year, 'file_name': file_name}

    def load_data(self, data_dir, SEED=42):

        filelist = glob.glob(data_dir+'/*.jpg')
        data = []

        # load just subset
        if self.subset is not None:
            filelist = itemgetter(*list(self.indices))(filelist)

        # iterate files and append to numpy array
        for i, filename in enumerate(tqdm(filelist, desc='Load Data')):
            im = Image.open(filename)
            data.append(np.array(im))

        return np.array(data)

    def get_name(self, idx, meta):
        key1 = 'celebrityData'
        cData = meta[key1][0, 0]
        # idx must be lowered by one because they are not 0 based
        if idx > 2000:
            raise ValueError('Idx must not exceed 2000')

        return cData[0][idx-1][0][0][0]

    def get_indices(self, data_dir, subset, SEED):

        filelist = glob.glob(data_dir+'/*.jpg')
        indices = list(range(len(filelist)))
        # shuffle
        np.random.seed(SEED)
        np.random.shuffle(indices)

        # take first subset values of filelist
        return indices[:subset]


    def __init__(self, meta_path, data_dir, subset=None, transform=None, SEED=42):
        """
        Args:s
            meta_path (string): Path to the .mat file with labels.
            data_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        #self.age = self.load_meta(meta_path)['age']
        #self.celeb_id = self.load_meta(meta_path)['celeb_id']
        #self.year = self.load_meta(meta_path)['year']
        #self.file_name = self.load_meta(meta_path)['file_name']
        self.subset = subset
        if subset is not None:
            self.indices = self.get_indices(data_dir, subset, SEED=SEED)
        self.transform = transform
        self.data_dir = data_dir
        self.meta_path = meta_path
        self.meta = sio.loadmat(meta_path)
        self.data = self.load_data(data_dir)
        self.meta_data = self.load_meta(meta_path)


    def __len__(self):
        return len(self.meta_data['file_name'])

    def __getitem__(self, idx):
        age = self.meta_data['age'][idx][0]
        name = self.get_name(self.meta_data['celeb_id'][idx], self.meta)
        im = self.data[idx]

        if self.transform:
            im = self.transform(im)

        sample = {'image': im, 'age': age, 'name': name}
        return sample


# function to resize images in data set
def resize_images(data_dir, target_dir, size=(32, 32)):
    """
    Args:
        data_dir (string): Directory with all the original images.
        target_dir (string): Directory the new images, will be created if not existant.
        size (tupel): Size of the output image.
    """
    filelist = glob.glob(data_dir+'/*.jpg')
    print('Filenames loaded...')

    # create new folder if not existant
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print('New Folder created!')
    else:
        print('Folder exists!')

    for i, filename in enumerate(tqdm(filelist, desc='Resize Images')):
        # get file
        im = Image.open(filename)
        # resize
        im.thumbnail(size, Image.ANTIALIAS)

        # outfile
        outfile = os.path.basename(filename)
        # save image
        im.save(target_dir+'/'+outfile)


if __name__ == '__main__':
    # set pathes to data
    meta_path = '../data/celebrity2000_meta.mat'
    data_dir = '../data/CACD2000'
    target_dir_32 = '../data/32x32CACD2000'
    target_dir_64 = '../data/64x64CACD2000'

    # explore_meta(meta_path)

    # data = load_images(data_dir)

    # test_dataset(meta_path, data_dir)

    resize_images(data_dir, target_dir_64, size=(64,64))
