import chainer
import numpy as np
import argparse
import sobamchan_utility
import sobamchan_iterator
import model
from tqdm import tqdm
import itertools

utility = sobamchan_utility.Utility()

def get_args():
    parser = argparse.ArgumentParser('gated convolutional networks trainer')
    parser.add_argument('--epoch', '-e', type=int, default=100, help='epoch size')
    parser.add_argument('--batch-size', '-b', type=int, default=128, help='batch size')
    parser.add_argument('--learning-rate', '-lr', type=float, default=0.1, help='initial learning rate')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    
    epoch = args.epoch
    batch_size = args.batch_size
    lr = args.learning_rate

    print('define model')
    resblock_n = 5
    in_channels = 1
    out_channels = 10
    ksize = 4
    category_n = 10
    model = model.Gated_Conv_Net(resblock_n, in_channels, out_channels, ksize, 10)
    optimizer = chainer.optimizers.MomentumSGD(lr, momentum=0.9)

    print('fake dataset')
    embedding_dim = 10
    sen_len = 1024
    train_x = np.random.rand(1000, 1, embedding_dim, sen_len)
    train_t = np.random.randint(2, size=1000)
    test_x = np.random.rand(100, 1, embedding_dim, sen_len)
    test_t = np.random.randint(2, size=100)

    train_N = len(train_x)

    for i in tqdm(range(epoch)):
        accum_loss = 0
        order = np.random.permutation(train_N)
        train_x_iter = sobamchan_iterator.Iterator(train_x, batch_size, order=order, shuffle=False)
        train_t_iter = sobamchan_iterator.Iterator(train_t, batch_size, order=order, shuffle=False)
        for x, t in tqdm(zip(train_x_iter, train_t_iter)):
            x = model.prepare_input(x, dtype=np.float32, xp=np)
            t = model.prepare_input(t, dtype=np.int32, xp=np)
            y = model(x)
