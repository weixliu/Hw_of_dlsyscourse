import numpy as np
from .autograd import Tensor
import struct
import gzip

from typing import Iterator, Optional, List, Sized, Union, Iterable, Any

def parse_mnist(image_filesname, label_filename):
    """ Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded
                data.  The dimensionality of the data should be
                (num_examples x input_dim) where 'input_dim' is the full
                dimension of the data, e.g., since MNIST images are 28x28, it
                will be 784.  Values should be of type np.float32, and the data
                should be normalized to have a minimum value of 0.0 and a
                maximum value of 1.0.

            y (numpy.ndarray[dypte=np.int8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.int8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR SOLUTION
    images = gzip.open(image_filesname, mode="rb")
    images_bytes = images.read()
    
    dimension = struct.unpack(">i", images_bytes[4:8])[0]
    row = struct.unpack(">i", images_bytes[8:12])[0]
    column = struct.unpack(">i", images_bytes[12:16])[0]
    
    idx = 16
    data_set = []
    for i in range(dimension):
        data = []
        for j in range(28):
            for k in range(28):
                # normalization
                data.append(struct.unpack(">B", images_bytes[idx:idx+1])[0] / 255)
                idx += 1
        data_set.append(data)
    image_res = np.array(data_set, dtype='f')
    # print(image_res.shape)
    
    labels = gzip.open(label_filename, mode="rb")
    labels_bytes = labels.read()
    dimension = struct.unpack(">i", images_bytes[4:8])[0]
    idx = 8
    data_set = []
    for i in range(dimension):
        data_set.append(struct.unpack(">B", labels_bytes[idx:idx+1])[0])
        idx += 1
    label_res = np.array(data_set, dtype='B')
    # print(label_res.shape)
    return (image_res, label_res)
    ### END YOUR SOLUTION

class Transform:
    def __call__(self, x):
        raise NotImplementedError


class RandomFlipHorizontal(Transform):
    def __init__(self, p = 0.5):
        self.p = p

    def __call__(self, img):
        """
        Horizonally flip an image, specified as n H x W x C NDArray.
        Args:
            img: H x W x C NDArray of an image
        Returns:
            H x W x C ndarray corresponding to image flipped with probability self.p
        Note: use the provided code to provide randomness, for easier testing
        """
        if(len(img.shape) == 2):
          img = np.expand_dims(img, axis=2)
        # print("img shape in [RandomFlipHorizontal]:{}".format(img.shape))
        flip_img = np.random.rand() < self.p
        ### BEGIN YOUR SOLUTION
        if(flip_img):
          img = np.flip(img, axis=1)
        return img
        ### END YOUR SOLUTION


class RandomCrop(Transform):
    def __init__(self, padding=3):
        self.padding = padding

    def __call__(self, img):
        """ Zero pad and then randomly crop an image.
        Args:
             img: H x W x C NDArray of an image
        Return 
            H x W x C NAArray of cliped image
        Note: generate the image shifted by shift_x, shift_y specified below
        """
        if(len(img.shape) == 2):
          img = np.expand_dims(img, axis=2)
        # print("img shape in [RandomCrop]:{}".format(img.shape))
        shift_x, shift_y = np.random.randint(low=-self.padding, high=self.padding+1, size=2)
        ### BEGIN YOUR SOLUTION
        x = img.shape[0]
        y = img.shape[1]
        init_x = self.padding + shift_x
        init_y = self.padding + shift_y
        return np.pad(img,
        ((self.padding,self.padding),(self.padding,self.padding),(0,0)))[init_x:init_x+x,init_y:init_y+y,:]
        ### END YOUR SOLUTION


class Dataset:
    r"""An abstract class representing a `Dataset`.

    All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Subclasses must also overwrite
    :meth:`__len__`, which is expected to return the size of the dataset.
    """

    def __init__(self, transforms: Optional[List] = None):
        self.transforms = transforms

    def __getitem__(self, index) -> object:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def apply_transforms(self, x):
        if self.transforms is not None:
            # apply the transforms
            for tform in self.transforms:
                x = tform(x)
        return x


class DataLoader:
    r"""
    Data loader. Combines a dataset and a sampler, and provides an iterable over
    the given dataset.
    Args:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: ``1``).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
     """
    dataset: Dataset
    batch_size: Optional[int]

    def __init__(
        self,
        dataset: Dataset,
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
    ):

        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
        if not self.shuffle:
            self.ordering = np.array_split(np.arange(len(dataset)), 
                                           range(batch_size, len(dataset), batch_size))

    def __iter__(self):
        ### BEGIN YOUR SOLUTION
        self.idx = 0
        idx_ndarray = np.arange(len(self.dataset))
        if self.shuffle:
          np.random.shuffle(idx_ndarray)
        self.ordering = np.array_split(idx_ndarray, range(self.batch_size, len(self.dataset), self.batch_size))
        ### END YOUR SOLUTION
        return self

    def __next__(self):
        ### BEGIN YOUR SOLUTION
        if(self.idx < len(self.ordering)):
          temp = self.dataset[self.ordering[self.idx]]
          self.idx += 1
          return tuple([Tensor.make_const(x) for x in temp])
        raise StopIteration
        ### END YOUR SOLUTION


class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        X, y = parse_mnist(image_filename, label_filename)
        self.X = X.reshape((X.shape[0], 28, 28, 1))
        self.y = y
        # print("self.X shape:{}".format(self.X.shape))
        # print("self.y shape:{}".format(self.y.shape))
        self.transforms = transforms
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        X = self.X[index]
        label = self.y[index]
        img = X
        if self.transforms != None:
          imgs = []
          for t_i in img:
            for t in self.transforms:
              t_i = t(t_i)
            imgs.append(t_i)
          img = np.array(imgs)
        return (img, label)
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return self.X.shape[0]
        ### END YOUR SOLUTION

class NDArrayDataset(Dataset):
    def __init__(self, *arrays):
        self.arrays = arrays

    def __len__(self) -> int:
        return self.arrays[0].shape[0]

    def __getitem__(self, i) -> object:
        return tuple([a[i] for a in self.arrays])
