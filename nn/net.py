#!/usr/bin/env python
import numpy

import chainer
import chainer.functions as F
import chainer.links as L
import chainerx


def add_noise(device, h, sigma=0.2):
    if chainer.config.train:
        xp = device.xp
        # TODO(niboshi): Support random.randn in ChainerX
        if device.xp is chainerx:
            fallback_device = device.fallback_device
            with chainer.using_device(fallback_device):
                randn = device.send(fallback_device.xp.random.randn(*h.shape))
        else:
            randn = xp.random.randn(*h.shape)
        return h + sigma * randn
    else:
        return h

def upsample_x2(x):
    _, c, h, w = x.shape
    x = F.reshape(x, (-1, 1))
    x = F.concat([x, x])
    x = F.reshape(x, (-1, w*2))
    x = F.concat([x, x])
    return F.reshape(x, (-1, c, h*2, w*2))
 
class UpsampleConvolution(chainer.Chain):

    def __init__(self, in_channels, out_channels, k=3, s=1, p=1, w=None):
        super().__init__()
        with self.init_scope():
            self.conv = L.Convolution2D(in_channels, out_channels, k, s, p, initialW=w)

    def forward(self, x):
        h = upsample_x2(x)
        return self.conv(h)

class PixelShuffler(chainer.Chain):
    """make spatial size 2time bigger and reduce channels to half
    """
    def __init__(self, in_channels, ksize=3, stride=1, pad=1, r=2, w=None):
        super().__init__()
        with self.init_scope():
            self.conv = L.Convolution2D(in_channels, in_channels * (r**2) // 2, ksize, stride, pad, initialW=w)
        self.r = r

    def forward(self, x):
        r = self.r
        out = self.conv(x)
        batchsize = out.shape[0]
        in_channels = out.shape[1]
        out_channels = in_channels // (r ** 2)
        in_height = out.shape[2]
        in_width = out.shape[3]
        out_height = in_height * r
        out_width = in_width * r
        out = F.reshape(out, (batchsize, r, r, out_channels, in_height, in_width))
        out = F.transpose(out, (0, 3, 4, 1, 5, 2))
        out = F.reshape(out, (batchsize, out_channels, out_height, out_width))
        return out

class Generator(chainer.Chain):

    def __init__(self, n_hidden, bottom_width=4, ch=512, wscale=0.02, use_pixelshuffler=False):
        super(Generator, self).__init__()
        self.n_hidden = n_hidden
        self.ch = ch
        self.bottom_width = bottom_width

        with self.init_scope():
            w = chainer.initializers.Normal(wscale)
            self.l0 = L.Linear(self.n_hidden, bottom_width * bottom_width * ch,
                               initialW=w)
            if use_pixelshuffler:
                print('using pixel shuffler')
                self.dc1 = PixelShuffler(ch, w=w)
                self.dc2 = PixelShuffler(ch//2, w=w)
                self.dc3 = PixelShuffler(ch//4, w=w)
                self.dc4 = PixelShuffler(ch//8, w=w)
                self.dc5 = UpsampleConvolution(ch//8, 3, w=w)
            else:
                print('using upsample and convlution')
                self.dc1 = UpsampleConvolution(ch, ch//2, w=w)
                self.dc2 = UpsampleConvolution(ch//2, ch//4, w=w)
                self.dc3 = UpsampleConvolution(ch//4, ch//8, w=w)
                self.dc5 = UpsampleConvolution(ch//8, 3, w=w)

            self.bn0 = L.BatchNormalization(bottom_width * bottom_width * ch)
            self.bn1 = L.BatchNormalization(ch // 2)
            self.bn2 = L.BatchNormalization(ch // 4)
            self.bn3 = L.BatchNormalization(ch // 8)
            self.bn4 = L.BatchNormalization(ch // 8)

    def make_hidden(self, batchsize):
        dtype = chainer.get_dtype()
        return numpy.random.uniform(-1, 1, (batchsize, self.n_hidden, 1, 1))\
            .astype(dtype)

    def forward(self, z):
        z = self.l0(z)
        z = F.reshape(z, (-1, self.ch * self.bottom_width*self.bottom_width, 1, 1))
        h = F.reshape(F.relu(self.bn0(z)),
                      (len(z), self.ch, self.bottom_width, self.bottom_width))
        h = F.relu(self.bn1(self.dc1(h)))
        h = F.relu(self.bn2(self.dc2(h)))
        h = F.relu(self.bn3(self.dc3(h)))
        x = F.sigmoid(self.dc5(h))
        return x


class Discriminator(chainer.Chain):

    def __init__(self, bottom_width=4, ch=512, wscale=0.02):
        w = chainer.initializers.Normal(wscale)
        super(Discriminator, self).__init__()
        with self.init_scope():
            self.c0_0 = L.Convolution2D(3, ch // 8, 3, 1, 1, initialW=w)
            self.c0_1 = L.Convolution2D(ch // 8, ch // 4, 4, 2, 1, initialW=w)
            self.c1_0 = L.Convolution2D(ch // 4, ch // 4, 3, 1, 1, initialW=w)
            self.c1_1 = L.Convolution2D(ch // 4, ch // 2, 4, 2, 1, initialW=w)
            self.c2_0 = L.Convolution2D(ch // 2, ch // 2, 3, 1, 1, initialW=w)
            self.c2_1 = L.Convolution2D(ch // 2, ch // 1, 4, 2, 1, initialW=w)
            self.c3_0 = L.Convolution2D(ch // 1, ch // 1, 3, 1, 1, initialW=w)
            self.l4 = L.Linear(None, 1, initialW=w)
            self.bn0_1 = L.BatchNormalization(ch // 4, use_gamma=False)
            self.bn1_0 = L.BatchNormalization(ch // 4, use_gamma=False)
            self.bn1_1 = L.BatchNormalization(ch // 2, use_gamma=False)
            self.bn2_0 = L.BatchNormalization(ch // 2, use_gamma=False)
            self.bn2_1 = L.BatchNormalization(ch // 1, use_gamma=False)
            self.bn3_0 = L.BatchNormalization(ch // 1, use_gamma=False)

    def forward(self, x):
        device = self.device
        h = add_noise(device, x)
        h = F.leaky_relu(add_noise(device, self.c0_0(h)))
        h = F.leaky_relu(add_noise(device, self.bn0_1(self.c0_1(h))))
        h = F.leaky_relu(add_noise(device, self.bn1_0(self.c1_0(h))))
        h = F.leaky_relu(add_noise(device, self.bn1_1(self.c1_1(h))))
        h = F.leaky_relu(add_noise(device, self.bn2_0(self.c2_0(h))))
        h = F.leaky_relu(add_noise(device, self.bn2_1(self.c2_1(h))))
        h = F.leaky_relu(add_noise(device, self.bn3_0(self.c3_0(h))))
        return self.l4(F.average_pooling_2d(h, h.shape[2]))
