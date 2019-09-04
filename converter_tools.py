from models.ofos import Net as TorchNet
import madmom.ml.nn.activations as act
import madmom.ml.nn.layers as mm
import matplotlib.pyplot as plt
from torch import nn
import numpy as np
import madmom
import torch


def plot_features(a, b):
    n_feature_maps = a.shape[-1]
    fig, axes = plt.subplots(
        nrows=n_feature_maps,
        ncols=3,
        sharex=True,
        sharey=True,
        squeeze=False
    )
    tr_v = np.max(np.abs(a))
    np_v = np.max(np.abs(b))
    v = max(tr_v, np_v)

    diff = a - b
    r = np.max(np.abs(diff))
    print('r', r)
    axes[0, 0].set_title('a')
    axes[0, 1].set_title('b')
    axes[0, 2].set_title('diff r {:>4.2g}'.format(r))

    for fi in range(n_feature_maps):
        axes[fi, 0].imshow(a[:, :, fi], cmap='seismic', vmin=-v, vmax=v)
        axes[fi, 1].imshow(b[:, :, fi], cmap='seismic', vmin=-v, vmax=v)
        axes[fi, 2].imshow(diff[:, :, fi], cmap='seismic', vmin=-r, vmax=r)

    plt.show()


def convert_conv(torch_conv, activation_fn):
    weights = torch_conv.weight.detach().numpy()
    bias = torch_conv.bias.detach().numpy()
    weights = np.transpose(weights, axes=(1, 0, 2, 3))

    # gotta flip them kernels!
    weights = np.flip(weights, 2)
    weights = np.flip(weights, 3)

    return mm.ConvolutionalLayer(weights, bias, activation_fn=activation_fn)


def convert_dense(torch_dense, activation_fn):
    weights = torch_dense.weight.t().detach().numpy()
    bias = torch_dense.bias.detach().numpy()
    return mm.FeedForwardLayer(weights, bias, activation_fn=activation_fn)


def convert_branch(sequential):
    layers = []

    for layer in sequential:
        if isinstance(layer, nn.Conv2d):
            layers.append(convert_conv(layer, act.elu))
            # we have to transpose the result of the conv layer
            # so when the data is reshaped, its in-mem order
            # fits to what the dense layer expects downstream
            layers.append(mm.TransposeLayer((0, 2, 1)))
            layers.append(mm.ReshapeLayer((-1, 1060)))
        if isinstance(layer, nn.Linear):
            layers.append(convert_dense(layer, act.sigmoid))
    return layers


def convert_batchnorm(torch_bn):
    beta = torch_bn.bias.detach().numpy()
    gamma = torch_bn.weight.detach().numpy()
    mean = torch_bn.running_mean.detach().numpy()
    var = torch_bn.running_var.detach().numpy()
    inv_std = 1. / np.sqrt(var + torch_bn.eps)
    print('beta', beta)
    print('gamma', gamma)
    print('mean', mean)
    print('var', var)
    return mm.BatchNormLayer(beta, gamma, mean, inv_std, act.linear)


def __get_random_input():
    C = 11
    tr_x = torch.Tensor(1, 1, C, 144).normal_(0, 6)
    np_x = tr_x.squeeze(0).numpy()
    np_x = np.transpose(np_x, (1, 2, 0))
    return tr_x, np_x


def __get_random_input_small():
    tr_x = torch.Tensor(1, 1, 9, 11).uniform_(-1, 1)
    np_x = tr_x.squeeze(0).numpy()
    np_x = np.transpose(np_x, (1, 2, 0))
    return tr_x, np_x


def __get_random_input_dense():
    tr_x = torch.Tensor(1, 11).uniform_(-1, 1)
    np_x = tr_x.numpy()
    return tr_x, np_x


def __test_conv():
    n_feature_maps = 5
    tr_x, np_x = __get_random_input_small()

    tr_conv = nn.Conv2d(1, n_feature_maps, (3, 3))
    mm_conv = convert_conv(tr_conv, act.linear)

    tr_y = tr_conv.forward(tr_x).squeeze().detach().numpy()
    tr_y = np.transpose(tr_y, (1, 2, 0))
    print('tr_y.shape', tr_y.shape)

    np_y = mm_conv.activate(np_x)
    print('np_y.shape', np_y.shape)

    print('np.allclose(tr_y, np_y)', np.allclose(tr_y, np_y))

    plot_features(tr_y, np_y)


def __test_bn():
    tr_x, np_x = __get_random_input_small()
    tr_bn = nn.BatchNorm2d(1)
    tr_bn.weight.data.uniform_(-1, 1)
    tr_bn.bias.data.uniform_(-1, 1)
    tr_bn.running_mean.data.uniform_(-1, 1)
    tr_bn.running_var.data.uniform_(1e-5, 1)
    tr_bn.training = False
    mm_bn = convert_batchnorm(tr_bn)

    tr_y = tr_bn.forward(tr_x).squeeze(0).detach().numpy()
    tr_y = np.transpose(tr_y, (1, 2, 0))
    print('tr_y.shape', tr_y.shape)
    np_y = mm_bn.activate(np_x)
    print('np_y.shape', np_y.shape)

    print('np.allclose(tr_y, np_y)', np.allclose(tr_y, np_y))

    plot_features(tr_y, np_y)


def __test_dense():
    tr_x, np_x = __get_random_input_dense()
    tr_dense = nn.Linear(11, 9)
    mm_dense = convert_dense(tr_dense, act.linear)

    tr_y = tr_dense.forward(tr_x).detach().numpy()
    np_y = mm_dense.activate(np_x)

    print('tr_y', tr_y)
    print('np_y', np_y)


def __test_model():
    torch_net = TorchNet(dict())
    print(torch_net)
    tr_x, np_x = __get_random_input()
    print('tr_x.size()', tr_x.size())
    print('np_x.shape', np_x.shape)
    torch_net.eval()
    with torch.no_grad():
        tr_y = torch_net.forward(dict(x=tr_x))

    bn = convert_batchnorm(torch_net.batch_norm)

    h_stem = []
    h_stem.append(bn)
    for layer in torch_net.conv_stem:
        if isinstance(layer, nn.Conv2d):
            h_stem.append(convert_conv(layer, act.elu))

    note_frames = convert_branch(torch_net.note_frames)
    note_onsets = convert_branch(torch_net.note_onsets)
    note_offsets = convert_branch(torch_net.note_offsets)

    out = madmom.processors.ParallelProcessor([note_frames, note_onsets, note_offsets])
    net = madmom.processors.SequentialProcessor(h_stem + [out])

    print('check net')
    np_y = net.process(np_x)
    print('np_y[0].shape', np_y[0].shape)
    print('np_y[1].shape', np_y[1].shape)
    print('np_y[2].shape', np_y[2].shape)

    for i, name in enumerate(['y_frames', 'y_onsets', 'y_offsets']):
        np_frames = np_y[i]
        tr_frames = tr_y[name].numpy()

        np_frames = np_frames.reshape((11, 8, 1))
        tr_frames = tr_frames.reshape((11, 8, 1))
        plot_features(tr_frames, np_frames)


def main():
    __test_conv()
    __test_bn()
    __test_dense()
    __test_model()


if __name__ == '__main__':
    main()
