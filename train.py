import sys
import numpy as np
import chainer
from chainer import cuda
import chainer.functions as F
import time

import utils


class Trainer:
    def __init__(self, model, optimizer, train_iter, val_iter, opt):
        self.model = model
        self.optimizer = optimizer
        self.train_iter = train_iter
        self.val_iter = val_iter
        self.opt = opt
        self.n_batches = (len(train_iter.dataset) - 1) // opt.batchSize + 1
        self.start_time = time.time()

    def train(self, epoch):
        self.optimizer.lr = self.lr_schedule(epoch)
        train_loss = 0
        train_acc = 0
        for i, batch in enumerate(self.train_iter):
            x_array, t_array = chainer.dataset.concat_examples(batch)
            x_array = np.reshape(x_array,(self.opt.batchSize*2,-1)).astype('float32')
            t_array = np.reshape(t_array,(self.opt.batchSize*2,-1)).astype('float32')
            x = chainer.Variable(cuda.to_gpu(x_array[:, None, None, :]))
            t = chainer.Variable(cuda.to_gpu(t_array))
            self.model.cleargrads()
            y , t = self.model(x, t, self.opt.mixup_type, self.opt.eligible, self.opt.batchSize)
            if self.opt.BC:
                loss = utils.kl_divergence(y, t)
                acc = F.accuracy(y, F.argmax(t, axis=1))
            else:
                loss = F.softmax_cross_entropy(y, t)
                acc = F.accuracy(y, t)

            loss.backward()
            self.optimizer.update()
            train_loss += float(loss.data) * len(t.data)
            train_acc += float(acc.data) * len(t.data)

            elapsed_time = time.time() - self.start_time
            progress = (self.n_batches * (epoch - 1) + i + 1) * 1.0 / (self.n_batches * self.opt.nEpochs)
            if ((progress)!=0):
                eta = elapsed_time / progress - elapsed_time
            else:
                eta = 0
            line = '* Epoch: {}/{} ({}/{}) | Train: LR {} | Time: {} (ETA: {})'.format(
                epoch, self.opt.nEpochs, i + 1, self.n_batches,
                self.optimizer.lr, utils.to_hms(elapsed_time), utils.to_hms(eta))
            sys.stderr.write('\r\033[K' + line)
            sys.stderr.flush()

        self.train_iter.reset()
        train_loss /= len(self.train_iter.dataset)*2
        train_top1 = 100 * (1 - train_acc / (len(self.train_iter.dataset)*2))

        return train_loss, train_top1

    def val(self):
        with chainer.using_config('train',False):
            val_acc = 0
            for batch in self.val_iter:
                x_array, t_array = chainer.dataset.concat_examples(batch)
                if self.opt.nCrops > 1:
                    x_array = x_array.reshape((x_array.shape[0] * self.opt.nCrops, x_array.shape[2]))
                x = chainer.Variable(cuda.to_gpu(x_array[:, None, None, :]))
                t = chainer.Variable(cuda.to_gpu(t_array))
                with chainer.no_backprop_mode():
                    y = F.softmax(self.model(x))
                    y = F.reshape(y, (y.shape[0] // self.opt.nCrops, self.opt.nCrops, y.shape[1]))
                    y = F.mean(y, axis=1)
                    acc = F.accuracy(y, t)
                    val_acc += float(acc.data) * len(t.data)

            self.val_iter.reset()
        val_top1 = 100 * (1 - val_acc / len(self.val_iter.dataset))

        return val_top1

    def lr_schedule(self, epoch):
        divide_epoch = np.array([self.opt.nEpochs * i for i in self.opt.schedule])
        decay = sum(epoch > divide_epoch)
        if epoch <= self.opt.warmup:
            decay = 1

        return self.opt.LR * np.power(0.1, decay)
