from torch.optim.lr_scheduler import _LRScheduler
import numpy as np


class OneCycleLR(_LRScheduler):
    """Sets the learning rate and momentum according to
       https://arxiv.org/abs/1803.09820

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        lrs (tuple): (start, middle, end)
        moms (tuple): (start, middle, end)
        n_epochs (int): the number of epochs (cycle length)
        last_epoch (int): when last_epoch == -1 sets lr_start as lr
    """

    def __init__(self, optimizer, lrs, moms, n_epochs, last_epoch=-1):
        self.optimizer = optimizer
        self.lr_start, self.lr_middle, self.lr_end = lrs
        self.mom_start, self.mom_middle, self.mom_end = moms
        self.n_epochs = n_epochs

        last = self.n_epochs // 10  # the last leg
        nep = self.n_epochs - last
        first = nep // 2
        second = nep - first

        self.lr_schedule = np.hstack([
            np.linspace(self.lr_start, self.lr_middle, first, endpoint=False),
            np.linspace(self.lr_middle, self.lr_start, second, endpoint=False),
            np.linspace(self.lr_start, self.lr_end, last)
        ])
        self.mom_schedule = np.hstack([
            np.linspace(self.mom_start, self.mom_middle, first, endpoint=False),
            np.linspace(self.mom_middle, self.mom_start, second, endpoint=False),
            np.linspace(self.mom_start, self.mom_end, last),
        ])

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [self.lr_schedule[int(self.last_epoch)]] * len(self.optimizer.param_groups)

    def get_mom(self):
        return [self.mom_schedule[int(self.last_epoch)]] * len(self.optimizer.param_groups)

    def step(self, metrics, epoch=None):
        if epoch is None:
            if hasattr(self, 'last_epoch'):
                epoch = self.last_epoch + 1
            else:
                epoch = 0

        self.last_epoch = epoch
        for param_group, lr, mom in zip(self.optimizer.param_groups, self.get_lr(), self.get_mom()):
            param_group['lr'] = lr
            param_group['momentum'] = mom
