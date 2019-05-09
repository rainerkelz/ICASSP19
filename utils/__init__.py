from torch.utils.data.sampler import Sampler
import numpy as np
from collections import defaultdict
from .wrapped_summary_writer import WrappedSummaryWriter


# can handle lists of nested dictionaries
def stacked_dict(list_of_dicts):
    dict_of_lists = defaultdict(list)
    for dictionary in list_of_dicts:
        for key, value in dictionary.items():
            if isinstance(value, dict):
                dict_of_lists[key] = stacked_dict(
                    (outer_dict[key] for outer_dict in list_of_dicts)
                )
            else:
                dict_of_lists[key].append(value)
    for key, value in dict_of_lists.items():
        if isinstance(value, list):
            dict_of_lists[key] = np.vstack(value)
    return dict(dict_of_lists)


def canonicalize_audio_options(_audio_options, mmspec):
    audio_options = dict(_audio_options)
    whitelisted_keys = set([
        'sample_rate',
        'frame_size',
        'fft_size',
        'hop_size',
        'num_channels',
        'spectrogram_type',
        'filterbank',
        'num_bands',
        'fmin',
        'fmax',
        'fref',
        'norm',
        'norm_filters',
        'unique_filters',
        'circular_shift'
    ])

    spectype = getattr(mmspec, audio_options['spectrogram_type'])
    del audio_options['spectrogram_type']

    if 'filterbank' in audio_options:
        audio_options['filterbank'] = getattr(mmspec, audio_options['filterbank'])

    # delete everything that is not in whitelist
    keys = list(audio_options.keys())
    for key in keys:
        if key not in whitelisted_keys:
            del audio_options[key]

    return spectype, audio_options


class ChunkedRandomSampler(Sampler):
    """Splits a dataset into smaller chunks (mainly to re-define what is considered an 'epoch').
       Samples elements randomly from a given list of indices, without replacement.
       If a chunk would be underpopulated, it's filled up with rest-samples.

    Arguments:
        data_source (Dataset): a dataset
        chunk_size      (int): how large a chunk should be
    """

    def __init__(self, data_source, chunk_size):
        self.data_source = data_source
        self.chunk_size = chunk_size
        self.i = 0
        self.N = len(self.data_source)
        # re-did this as numpy permutation, b/c FramedSignals do not like
        # torch tensors as indices ...
        # self.perm = torch.randperm(self.N)
        self.perm = np.random.permutation(self.N)

    def __iter__(self):
        rest = len(self.perm) - (self.i + self.chunk_size)
        if rest == 0:
            self.i = 0
            self.perm = np.random.permutation(self.N)
        elif rest < 0:
            # works b/c rest is negative
            carryover = self.chunk_size + rest
            self.i = 0
            self.perm = np.hstack([self.perm[-carryover:], np.random.permutation(self.N)])

        chunk = self.perm[self.i: self.i + self.chunk_size]
        self.i += self.chunk_size
        return iter(chunk)

    def __len__(self):
        return self.chunk_size


# from http://joseph-long.com/writing/colorbars/
def colorbar(mappable):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    return fig.colorbar(mappable, cax=cax)


def minmax(x, symm):
    if symm:
        r = np.max(np.abs(x))
        vmin = -r
        vmax = r
    else:
        vmin = np.min(x)
        vmax = np.max(x)
    return vmin, vmax


def flip(x, preference):
    rows, cols = x.shape
    if preference == 'horizontal':
        if rows > cols:
            return x.T
    else:
        if rows < cols:
            return x.T
    return x


def tensor_plot_helper(fig, ax, x, cmap, symm, origin='lower', interpolation='nearest', preference='horizontal'):
    if len(x.shape) == 1:
        vmin, vmax = minmax(x, symm)
        x = x.reshape(-1, 1)
        im = ax.imshow(flip(x, preference), cmap=cmap, vmin=vmin, vmax=vmax, origin=origin, interpolation=interpolation)
    if len(x.shape) == 2:
        w, h = x.shape
        vmin, vmax = minmax(x, symm)
        im = ax.imshow(flip(x, preference), cmap=cmap, vmin=vmin, vmax=vmax, origin=origin, interpolation=interpolation)
    elif len(x.shape) == 3:
        n, w, h = x.shape
        a = b = int(np.ceil(np.sqrt(n)))

        ai = 0
        matrices = []
        for i in range(a):
            row = []
            for j in range(b):
                if ai < n:
                    cell = np.pad(x[ai], (1,), mode='constant', constant_values=0.)
                    ai += 1
                else:
                    cell = np.zeros((w + 2, h + 2))
                row.append(cell)
            matrices.append(row)

        _x = np.bmat(matrices)
        vmin, vmax = minmax(_x, symm)
        im = ax.imshow(flip(_x, preference), cmap=cmap, vmin=vmin, vmax=vmax, origin=origin, interpolation=interpolation)
    elif len(x.shape) == 4:
        a, b, w, h = x.shape
        matrices = []
        for i in range(a):
            row = []
            for j in range(b):
                row.append(np.pad(x[i, j], (1,), mode='constant', constant_values=0.))
            matrices.append(row)

        _x = np.bmat(matrices)
        vmin, vmax = minmax(x, symm)
        im = ax.imshow(flip(_x, preference), cmap=cmap, vmin=vmin, vmax=vmax, origin=origin, interpolation=interpolation)

    return im


# WARNING! this is not the same truncated normal as in numpy and tensorflow!!!
def truncated_normal_(tensor, mean=0, std=1):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
