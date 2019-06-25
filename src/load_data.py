import numpy as np
import scipy.io as sio
from PIL import Image
import glob
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import os

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
def return_name(idx):
    meta = sio.loadmat(meta_path)
    key1 = 'celebrityData'
    cData = meta[key1][0, 0]
    return cData[0][idx][0][0]


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
        # load meta data into numpy array
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

        return {'age': age, 'celeb_id': celeb_id, 'year': year, 'file_name':file_name}

    def load_data(self, data_dir):
        filelist = glob.glob(relative_path+'/*.jpg')
        data = np.empty(len(filelist))

        # iterate files and append to numpy array
        print(len(filelist))
        for i, filename in enumerate(filelist):
            im = Image.open(filename)
            data[i] = np.array(im)
        return data

    def __init__(self, meta_path, data_dir, transform=None):
        """
        Args:
            meta_path (string): Path to the .mat file with labels.
            data_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.age = self.load_meta(meta_path)['age']
        self.celeb_id = self.load_meta(meta_path)['celeb_id']
        self.year = self.load_meta(meta_path)['year']
        self.file_name = self.load_meta(meta_path)['file_name']
        self.transform = transform
        self.data_dir = data_dir

    def __len__(self):
        return len(self.file_name)

    def __getitem__(self, idx):
        filename = os.path.join(self.data_dir,
                                self.file_name[idx][0][0])

        age = self.age[idx][0]
        name = return_name(self.celeb_id[idx])
        im = np.array(Image.open(filename))
        sample = {'image': im, 'age': age, 'name': name}

        if self.transform:
            sample = self.transform(sample)

        return sample




if __name__ == '__main__':
    # set pathes to data
    meta_path = '../data/celebrity2000_meta.mat'
    data_dir = '../data/CACD2000'

    # explore_meta(meta_path)

    # data = load_images(data_dir)

    test_dataset(meta_path, data_dir)
