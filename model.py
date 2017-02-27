import numpy as np
import sobamchan_chainer
import chainer.links as L
import chainer.functions as F
from chainer import Variable
import chainer

class GLU_conv(sobamchan_chainer.Model):

    def __init__(self, in_channels, out_channels, ksize):
        super(GLU_conv, self).__init__(
            conv=L.Convolution2D(in_channels, out_channels, (1, ksize)),
        )
        self.padding = int(ksize/2)

    @staticmethod
    def zero_padding(x, padding_size, axis=3):
        size = list(x.shape)
        size[axis] = padding_size
        zeros = Variable(np.zeros(size, dtype=np.float32))
        x = F.concat([zeros, x], axis)
        return x

    def __call__(self, x):
        x = self.zero_padding(x, self.padding)
        return self.conv(x)


class Gated_Unit(sobamchan_chainer.Model):

    def __init__(self, in_channels, out_channels, ksize):
        super(Gated_Unit, self).__init__(
            conv=GLU_conv(in_channels, out_channels, ksize),
            conv_g=GLU_conv(in_channels, out_channels, ksize)
        )

    def __call__(self, x):
        A = self.conv(x)
        B = F.sigmoid(self.conv_g(x))
        h = A * B
        batch, channel, height, width = h.shape
        return h

class ResBlock(sobamchan_chainer.Model):

    def __init__(self, block_n, in_channels, out_channels, ksize):
        super(ResBlock, self).__init__()
        modules = []
        for i in range(block_n):
            modules += [('gated_unit_{}'.format(i), Gated_Unit(in_channels, out_channels, ksize))]
            in_channels = out_channels
        [self.add_link(*link) for link in modules]
        self.modules = modules
        self.block_n = block_n
        
    def __call__(self, x, train=False):
        h = x
        for i in range(self.block_n):
            h = self['gated_unit_{}'.format(i)](h)
        channel_diff = h.shape[1] - x.shape[1]
        x = GLU_conv.zero_padding(x, channel_diff, 1)
        length_diff = x.shape[3] - h.shape[3]
        h = GLU_conv.zero_padding(h, length_diff, 3)
        return x + h


class Gated_Conv_Net(sobamchan_chainer.Model):

    def __init__(self, resblock_n, in_channels, out_channels, ksize, category_n):
        super(Gated_Conv_Net, self).__init__()
        modules = []
        for i in range(resblock_n):
            modules += [('resblock_{}'.format(i), ResBlock(3, in_channels, out_channels, ksize))]
            in_channels = out_channels
        modules += [('fc', L.Linear(None, category_n))]
        [ self.add_link(*link) for link in modules ]
        self.modules = modules
        self.resblock_n = resblock_n
        self.category_n = category_n

    def __call__(self, x, train=True):
        for i in range(self.resblock_n):
            x = self['resblock_{}'.format(i)](x, train)
        x = self['fc'](x)
        return x
        # batch = x.shape[0]
        # return F.reshape(x, (batch, self.category_n, -1))

    def cal_loss(self, y, t):
        return F.softmax_cross_entropy(y, t)

    def cal_acc(self, y, t):
        return F.accuracy(y, t)
