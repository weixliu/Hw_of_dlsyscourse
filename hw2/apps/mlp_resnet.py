import sys
sys.path.append('../python')
import needle as ndl
import needle.nn as nn
import needle.data as data
import numpy as np
import time
import os
import logging
import tracemalloc

LOGGER = logging.getLogger(__name__)

np.random.seed(0)

def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    sequential = []
    sequential.append(nn.Linear(dim, hidden_dim))
    sequential.append(norm(hidden_dim))
    sequential.append(nn.ReLU())
    sequential.append(nn.Dropout(p=drop_prob))
    sequential.append(nn.Linear(hidden_dim, dim))
    sequential.append(norm(dim))
    return nn.Sequential(
      nn.Residual(
        nn.Sequential(*sequential)
      ),nn.ReLU()
    )
    return nn.Sequential(*sequential)
    ### END YOUR SOLUTION


def MLPResNet(dim, hidden_dim=100, num_blocks=3, num_classes=10, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    print(norm)
    sequential = []
    sequential.append(nn.Linear(dim, hidden_dim))
    sequential.append(nn.ReLU())
    for i in range(num_blocks):
      sequential.append(ResidualBlock(hidden_dim, hidden_dim // 2, norm, drop_prob))
    sequential.append(nn.Linear(hidden_dim, num_classes))
    return nn.Sequential(*sequential)
    ### END YOUR SOLUTION

def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    flatten = nn.Flatten()
    softmaxLoss = nn.SoftmaxLoss()
    losses = 0.0
    err = 0.0
    total = 0.0
    if (opt == None):
        model.eval()
    else:
        model.train()
    for i, batch in enumerate(dataloader):
        x = batch[0]
        y = batch[1]
        total += x.shape[0]

        stime = time.perf_counter()

        if (opt != None):
            opt.reset_grad()

        x = flatten(x)
        pred_y = model(x)
        loss = softmaxLoss(pred_y, y)

        err += np.sum(pred_y.data.numpy().argmax(axis=1) != y.numpy())
        losses += loss.data.numpy() * x.shape[0]

        if (opt != None):
            # snapshot1 = tracemalloc.take_snapshot()

            loss.backward()

            # snapshot2 = tracemalloc.take_snapshot()
            # top_stats = snapshot2.compare_to(snapshot1, 'lineno')
            # LOGGER.warning("[ Top 50 differences in backward ]")
            # for stat in top_stats[:50]:
            #     LOGGER.warning(stat)

            opt.step()

            # snapshot3 = tracemalloc.take_snapshot()
            # top_stats = snapshot3.compare_to(snapshot2, 'lineno')
            # LOGGER.warning("[ Top 50 differences in step ]")
            # for stat in top_stats[:50]:
            #     LOGGER.warning(stat)
        LOGGER.info('mini batch takes {:.6f}s'.format(time.perf_counter() - stime))

    return err / total, losses / total
    ### END YOUR SOLUTION



def train_mnist(batch_size=100, epochs=10, optimizer=ndl.optim.Adam,
                lr=0.001, weight_decay=0.001, hidden_dim=100, data_dir="data"):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    train_x = data_dir + "/train-images-idx3-ubyte.gz"
    train_y = data_dir + "/train-labels-idx1-ubyte.gz"
    trainDataset = data.MNISTDataset(train_x, train_y)
    trainDataloader = data.DataLoader(trainDataset, batch_size, True)

    test_x = data_dir + "/t10k-images-idx3-ubyte.gz"
    test_y = data_dir + "/t10k-labels-idx1-ubyte.gz"
    testDataset = data.MNISTDataset(test_x, test_y)
    testDataloader = data.DataLoader(testDataset, batch_size)

    model = MLPResNet(784, hidden_dim)
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_tuple = None
    eval_tuple = None

    for i in range(epochs):
      # stats_memory()
      train_tuple = epoch(trainDataloader, model, opt)
      eval_tuple = epoch(testDataloader, model)
      LOGGER.warn("The epoch:{}, \n\ttrain acc:{}, \n\ttrain loss:{}, \n\ttest acc:{}, \n\ttest loss:{}" \
      .format(i, train_tuple[0], train_tuple[1], eval_tuple[0], eval_tuple[1]))
    return (train_tuple[0], train_tuple[1], eval_tuple[0], eval_tuple[1])
    ### END YOUR SOLUTION


def stats_memory():
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics('lineno')

    print("[ Top 10 ]")
    for stat in top_stats[:10]:
        print(stat)
        for line in stat.traceback.format():
            print(line)


if __name__ == "__main__":
    tracemalloc.start()

    # train_mnist(data_dir="../data")
    np.random.seed(1)
    out = train_mnist(250, 2, ndl.optim.SGD, 0.001, 0.01, 100, data_dir="../data")
