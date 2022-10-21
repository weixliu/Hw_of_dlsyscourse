import struct
import gzip
import numpy as np

import sys
sys.path.append('python/')
import needle as ndl
import time
import logging

LOGGER = logging.getLogger(__name__)

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
    
    labels = gzip.open(label_filename, mode="rb")
    labels_bytes = labels.read()
    dimension = struct.unpack(">i", images_bytes[4:8])[0]
    idx = 8
    data_set = []
    for i in range(dimension):
        data_set.append(struct.unpack(">B", labels_bytes[idx:idx+1])[0])
        idx += 1
    label_res = np.array(data_set, dtype='B')
    return (image_res, label_res)
    ### END YOUR SOLUTION


def softmax_loss(Z, y_one_hot):
    """ Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (ndl.Tensor[np.float32]): 2D Tensor of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (ndl.Tensor[np.int8]): 2D Tensor of shape (batch_size, num_classes)
            containing a 1 at the index of the true label of each example and
            zeros elsewhere.

    Returns:
        Average softmax loss over the sample. (ndl.Tensor[np.float32])
    """
    ### BEGIN YOUR SOLUTION
    # need to recall the lectures to understand more
    Z_minus = (y_one_hot * Z).sum(axes=(1,))
    return (ndl.log(ndl.exp(Z).sum(axes=(1,))) - Z_minus).sum() / Z.shape[0]
    ### END YOUR SOLUTION


def nn_epoch(tX, ty, W1, W2, lr = 0.1, batch=100):
    """ Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W1
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (ndl.Tensor[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (ndl.Tensor[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD mini-batch

    Returns:
        Tuple: (W1, W2)
            W1: ndl.Tensor[np.float32]
            W2: ndl.Tensor[np.float32]
    """

    ### BEGIN YOUR SOLUTION
    start = 0
    total = tX.shape[0]
    #LOGGER.info('The batch size:{}'.format(total))
    
    while(start < total):
      stime = time.perf_counter()
      X = tX[start:start + batch]
      y = ty[start:start + batch]
      y_one_hot = np.zeros((y.shape[0], W2.shape[1]))
      y_one_hot[np.arange(y.size), y] = 1
      y = ndl.Tensor(y_one_hot)
      Z = ndl.relu(ndl.Tensor(X).matmul(W1)).matmul(W2)
      loss = softmax_loss(Z, y)
      #LOGGER.info('{:.6f}s for the prepare the batch'.format(time.perf_counter() - stime))
      loss.backward()
      #LOGGER.info('{:.6f}s for compute gradient'.format(time.perf_counter() - stime))
      gradientW1 = W1.grad.detach()
      gradientW2 = W2.grad.detach()
      W1 -= lr * gradientW1
      W2 -= lr * gradientW2
      start += batch
    return (W1, W2)
    ### END YOUR SOLUTION


### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT

def loss_err(h,y):
    """ Helper function to compute both loss and error"""
    y_one_hot = np.zeros((y.shape[0], h.shape[-1]))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = ndl.Tensor(y_one_hot)
    return softmax_loss(h,y_).numpy(), np.mean(h.numpy().argmax(axis=1) != y)
