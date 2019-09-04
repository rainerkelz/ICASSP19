from models.ofos import Net as TorchNet
import madmom.ml.nn.activations as act
import madmom
import argparse
from converter_tools import convert_conv, convert_batchnorm, convert_branch
import torch
from torch import nn
import numpy as np

from madmom.processors import SequentialProcessor


def convert_model(torch_net, torch_weights):
    print(torch_net)
    torch_net.load_state_dict(torch_weights)

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
    net = madmom.processors.SequentialProcessor(h_stem + [out, np.dstack])

    return net


def _cnn_pad(data):
    """Pad the data by repeating the first and last frame 5 times."""
    pad_start = np.repeat(data[:1], 5, axis=0)
    pad_stop = np.repeat(data[-1:], 5, axis=0)
    return np.concatenate((pad_start, data, pad_stop))


def build_cnn(madmom_processor_filename):
    from madmom.audio.signal import SignalProcessor, FramedSignalProcessor
    from madmom.audio.stft import ShortTimeFourierTransformProcessor
    from madmom.audio.spectrogram import (FilteredSpectrogramProcessor,
                                          LogarithmicSpectrogramProcessor)

    from madmom.ml.nn import NeuralNetworkEnsemble
    # define pre-processing chain
    sig = SignalProcessor(num_channels=1, sample_rate=44100)
    frames = FramedSignalProcessor(frame_size=4096, hop_size=441 * 2)
    stft = ShortTimeFourierTransformProcessor()  # caching FFT window
    filt = FilteredSpectrogramProcessor(num_bands=24, fmin=30, fmax=10000)

    # this is the money param! it was not whitelisted in 'canonicalize_audio_options'!
    spec = LogarithmicSpectrogramProcessor(add=1)
    # pre-processes everything sequentially
    pre_processor = SequentialProcessor([
        sig, frames, stft, filt, spec, _cnn_pad
    ])
    # process the pre-processed signal with a NN
    nn = NeuralNetworkEnsemble.load([madmom_processor_filename])
    return madmom.processors.SequentialProcessor([pre_processor, nn])


def plot_output(data, title):
    import matplotlib.pyplot as plt
    print('data.shape', data.shape)

    fig, axes = plt.subplots(nrows=3, sharex=True, sharey=True)
    fig.suptitle(title)
    for i in range(3):
        axes[i].imshow(data[:, :, i].T, cmap='magma', vmin=0, vmax=1)

    data = data.reshape(-1, 88 * 3)
    fig, ax = plt.subplots()
    fig.suptitle(title)
    ax.imshow(data.T, cmap='magma', vmin=0, vmax=1)


def main():
    parser = argparse.ArgumentParser(description='integration testing')
    parser.add_argument('torch_weights_filename')
    parser.add_argument('madmom_processor_filename')
    parser.add_argument('--audio_filename',
                        default=None,
                        help='used for optional integration test')
    args = parser.parse_args()

    torch_weights = torch.load(args.torch_weights_filename)['net_state_dict']

    for key, value in torch_weights.items():
        print('key', key)

    net = convert_model(TorchNet(dict()), torch_weights)
    net.dump(args.madmom_processor_filename)

    if args.audio_filename is not None:
        import matplotlib.pyplot as plt
        cnn = build_cnn(args.madmom_processor_filename)
        data = cnn.process(args.audio_filename)
        plot_output(data, 'integrated integration test')
        plt.show()


if __name__ == '__main__':
    main()
