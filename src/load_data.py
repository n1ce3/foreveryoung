import numpy as np
import scipy.io as sio

def explore_meta(meta):

    print(type(meta))
    print(meta.keys())
    for key in meta.keys():
        print(key, type(meta[key]))

    # Keys which store data
    key1 = 'celebrityData'
    key2 = 'celebrityImageData'

    # Shape of numpy arrays
    print(key1, np.shape(meta[key1]))
    print(key2, np.shape(meta[key2]))

    # Data seems to be in the [0, 0][0] entry
    print(key1, np.shape(meta[key1][0, 0][0]))
    print(key2, np.shape(meta[key2][0, 0][0]))

    cData = meta[key1][0, 0][0]
    cImageData = meta[key2][0, 0][0]

    print(np.shape(cData))

    for i in range(np.shape(cData)[0]):
        print(i, cData[i, 0][0])

    # Look at one entry of cImageData
    print(np.shape(cImageData[0, 0]), cImageData[0, 0])

    # cImageData seems to contain reference to celebrity number in picture
    
if __name__ == '__main__':
    meta = sio.loadmat('../data/celebrity2000_meta.mat')
    explore_meta(meta)
