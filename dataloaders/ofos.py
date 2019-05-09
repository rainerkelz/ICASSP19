from torch.utils.data import Dataset, ConcatDataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from madmom.audio.signal import FramedSignal
import madmom.audio.spectrogram as mmspec
from madmom.io import midi
from scipy.ndimage.filters import maximum_filter1d
from copy import deepcopy
import numpy as np
import joblib
import torch
import utils
import os
import csv
import warnings


memory = joblib.memory.Memory('./joblib_cache', mmap_mode='r', verbose=1)


def get_y_from_file(midifile, n_frames, audio_options):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        pattern = midi.MIDIFile(midifile)
        dt = float(audio_options['hop_size']) / float(audio_options['sample_rate'])

        y_onsets = np.zeros((n_frames, 88), dtype=np.uint8)
        y_frames = np.zeros((n_frames, 88), dtype=np.uint8)
        y_offsets = np.zeros((n_frames, 88), dtype=np.uint8)

        for onset, _pitch, duration, velocity, _channel in pattern.sustained_notes:
            pitch = int(_pitch)
            label = pitch - 21

            note_start = int(np.round(onset / dt))
            note_end = int(np.round((onset + duration) / dt))

            # some of the midi-files have onsets/offsets larger
            # than n_frames. they were manually checked, and it's
            # actually not an issue at all.
            # see data-preparation/maestro-inconsistencies/* for
            # scripts that perform visual inspection!
            if note_start < n_frames:
                if note_end >= n_frames:
                    # print('weird_offset', midifile)
                    note_end = n_frames - 1

                y_onsets[note_start, label] = 1
                y_frames[note_start:note_end + 1, label] = 1
                y_offsets[note_end, label] = 1
            else:
                # print('weird_onset', midifile)
                pass

        return y_onsets, y_frames, y_offsets


def get_existing_filename(filename, suffixes):
    for suffix in suffixes:
        if os.path.exists(filename + suffix):
            return filename + suffix

    raise ValueError('provided filename "{}" does not exist with any of the endings {}'.format(filename, suffixes))


@memory.cache
def get_xy_from_file(audio_filename, midi_filename, _audio_options):
    spec_type, audio_options = utils.canonicalize_audio_options(_audio_options, mmspec)
    x = np.array(spec_type(audio_filename, **audio_options))
    y_onsets, y_frames, y_offsets = get_y_from_file(midi_filename, len(x), audio_options)

    return x, y_onsets, y_frames, y_offsets


def get_xy_from_file_subsampled(audio_filename, midi_filename, audio_options, start_end):
    x, y_onsets, y_frames, y_offsets = get_xy_from_file(
        audio_filename,
        midi_filename,
        audio_options
    )
    if start_end is None:
        return x, y_onsets, y_frames, y_offsets
    else:
        start, end = start_end
        return x[start:end], y_onsets[start:end], y_frames[start:end], y_offsets[start:end]


# maxfilter in temporal direction
def widen(x, w):
    if w % 2 == 0:
        raise RuntimeError('unsupported')

    return maximum_filter1d(x, w, axis=0, mode='constant', cval=0, origin=0)


def suppress_offets(y_onsets, y_offsets):
    # everywhere where onsets and offsets DO NOT occur simultaenously,
    # (y_onsets != y_offsets) will be False
    # only where onsets and offsets DO NOT overlap, it'll be True
    return (y_onsets != y_offsets) * 1


class OneSequenceDataset(Dataset):
    def __init__(self,
                 audio_filename,
                 midi_filename,
                 input_context,
                 target_maxfilter,
                 audio_options,
                 start_end=None,
                 offset_suppression=None):
        self.metadata = dict(
            audio_filename=audio_filename,
            midi_filename=midi_filename
        )
        self.audio_options = deepcopy(audio_options)

        x, y_onsets, y_frames, y_offsets = get_xy_from_file_subsampled(
            self.metadata['audio_filename'],
            self.metadata['midi_filename'],
            self.audio_options,
            start_end
        )

        self.y_onsets = widen(y_onsets, target_maxfilter['y_onsets'])
        self.y_frames = widen(y_frames, target_maxfilter['y_frames'])

        if offset_suppression is not None:
            # this gets passed the widened *onsets* already
            y_offsets = suppress_offets(y_onsets, y_offsets)

        # widen *after* suppression
        self.y_offsets = widen(y_offsets, target_maxfilter['y_offsets'])

        self.x = FramedSignal(
            x,
            frame_size=input_context['frame_size'],
            hop_size=input_context['hop_size'],
            origin=input_context['origin'],
        )
        if (len(self.x) != len(self.y_onsets) or
           len(self.x) != len(self.y_frames) or
           len(self.x) != len(self.y_offsets)):
            raise RuntimeError('x and y do not have the same length.')

    def __getitem__(self, index):
        _, w, h = self.x.shape
        return dict(
            x=torch.FloatTensor(self.x[index].reshape(1, w, h)),
            y_onsets=torch.FloatTensor(self.y_onsets[index]),
            y_frames=torch.FloatTensor(self.y_frames[index]),
            y_offsets=torch.FloatTensor(self.y_offsets[index])
        )

    def __len__(self):
        return len(self.x)


def get_dataset_individually(base_directory, metadata_filename, split, input_context, target_maxfilter, audio_options, start_end=None, offset_suppression=None):

    fieldnames = [
        'canonical_composer',
        'canonical_title',
        'split',
        'year',
        'midi_filename',
        'audio_filename',
        'duration'
    ]

    class magenta_dialect(csv.Dialect):
        delimiter = ','
        quotechar = '"'
        doublequote = True
        skipinitialspace = True
        lineterminator = '\n'
        quoting = csv.QUOTE_MINIMAL
    csv.register_dialect('magenta', magenta_dialect)

    sequences = []
    with open(metadata_filename, 'r') as metadata_file:
        csvreader = csv.DictReader(metadata_file, fieldnames=fieldnames, dialect='magenta')
        for row in csvreader:
            if row['split'] == split:
                sequences.append(OneSequenceDataset(
                    os.path.join(base_directory, row['audio_filename']),
                    os.path.join(base_directory, row['midi_filename']),
                    input_context,
                    target_maxfilter,
                    audio_options,
                    start_end,
                    offset_suppression
                ))
    return sequences


def get_dataset(*args, **kwargs):
    dataset = ConcatDataset(get_dataset_individually(*args, **kwargs))
    print('len(dataset)', len(dataset))
    return dataset


def get_loaders(config):
    lcs = config['modules']['dataloader']['args']
    loaders = dict()
    for key, lc in lcs.items():
        individual_files = lc.get('individual_files', False)
        if individual_files:
            sequences = get_dataset_individually(
                base_directory=lc['base_directory'],
                metadata_filename=lc['metadata_filename'],
                split=lc['split'],
                input_context=config['modules']['dataloader']['input_context'],
                target_maxfilter=config['modules']['dataloader']['target_maxfilter'],
                audio_options=config['audio_options'],
                start_end=config['modules']['dataloader'].get('start_end', None),
                offset_suppression=config['modules']['dataloader'].get('offset_suppression', None)
            )
            individual_loaders = []
            for sequence in sequences:
                if lc['sampler'] == 'RandomSampler':
                    sampler = RandomSampler(sequence)
                elif lc['sampler'] == 'SequentialSampler':
                    sampler = SequentialSampler(sequence)
                elif lc['sampler'] == 'ChunkedRandomSampler':
                    sampler = utils.ChunkedRandomSampler(sequence, lc['chunk_size'])

                individual_loader = DataLoader(
                    sequence,
                    batch_size=config['batchsize'],
                    sampler=sampler,
                    num_workers=lc.get('num_workers', 0),
                    pin_memory=lc.get('pin_memory', False)
                )
                individual_loaders.append(individual_loader)
            loaders[key] = individual_loaders
        else:
            sequences = get_dataset(
                base_directory=lc['base_directory'],
                metadata_filename=lc['metadata_filename'],
                split=lc['split'],
                input_context=config['modules']['dataloader']['input_context'],
                target_maxfilter=config['modules']['dataloader']['target_maxfilter'],
                audio_options=config['audio_options'],
                start_end=config['modules']['dataloader'].get('start_end', None),
                offset_suppression=config['modules']['dataloader'].get('offset_suppression', None)
            )

            if lc['sampler'] == 'RandomSampler':
                sampler = RandomSampler(sequences)
            elif lc['sampler'] == 'SequentialSampler':
                sampler = SequentialSampler(sequences)
            elif lc['sampler'] == 'ChunkedRandomSampler':
                sampler = utils.ChunkedRandomSampler(sequences, lc['chunk_size'])

            loader = DataLoader(
                sequences,
                batch_size=config['batchsize'],
                sampler=sampler,
                num_workers=lc.get('num_workers', 0),
                pin_memory=lc.get('pin_memory', False)
            )

            loaders[key] = loader
    return loaders


def main():
    context = dict(
        frame_size=5,
        hop_size=1,
        origin='center'
    )

    target_maxfilter = dict(
        y_onsets=3,
        y_frames=1,
        y_offsets=3
    )

    audio_options = dict(
        spectrogram_type='LogarithmicFilteredSpectrogram',
        filterbank='LogarithmicFilterbank',
        sample_rate=44100,
        num_channels=1,
        frame_size=2048,
        hop_size=1024
    )

    sequences = get_dataset(
        base_directory='./data/maestro/maestro-v1.0.0/',
        metadata_filename='./data/maestro/maestro-v1.0.0/maestro-v1.0.0.csv',
        split='train',
        input_context=context,
        target_maxfilter=target_maxfilter,
        audio_options=audio_options
    )
    print('len(sequences)', len(sequences))

    loader = DataLoader(
        sequences,
        batch_size=128,
        shuffle=False,
        sampler=SequentialSampler(sequences),
        drop_last=False
    )

    import matplotlib.pyplot as plt

    batch = next(iter(loader))
    x = batch['x'].numpy()[:, 0, 2, :]
    y_onsets = batch['y_onsets'].numpy()
    y_frames = batch['y_frames'].numpy()
    y_offsets = batch['y_offsets'].numpy()

    fig, axes = plt.subplots(nrows=4, sharex=True, sharey=True)

    axes[0].set_title('x')
    axes[1].set_title('y_onsets')
    axes[2].set_title('y_frames')
    axes[3].set_title('y_offsets')

    axes[0].imshow(x.T, origin='lower')
    axes[1].imshow(y_onsets.T, cmap='gray_r', origin='lower')
    axes[2].imshow(y_frames.T, cmap='gray_r', origin='lower')
    axes[3].imshow(y_offsets.T, cmap='gray_r', origin='lower')
    plt.show()


if __name__ == '__main__':
    main()
