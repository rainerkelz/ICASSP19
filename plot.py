import torch
import argparse
import numpy as np
import mir_eval
from madmom.io import midi
from adsr import ADSRNoteTrackingProcessor
from matplotlib.collections import PatchCollection
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
np.set_printoptions(precision=4)


def get_onsets_and_pitch_labels(midifile):
    pattern = midi.MIDIFile(midifile)
    intervals = []
    labels = []
    for onset, _pitch, duration, velocity, _channel in pattern.sustained_notes:
        label = int(_pitch)  # do not subtract 21; mir_eval needs pitches strictly >= 0 anyways
        intervals.append([onset, onset + duration])
        labels.append(label)
    return np.array(intervals), np.array(labels)


def evaluate(est_intervals, est_pitches, ref_intervals, ref_pitches):
    if len(est_intervals) > 0:
        # evaluate onsets and pitches
        p, r, f, o = mir_eval.transcription.precision_recall_f1_overlap(
            ref_intervals,
            ref_pitches,
            est_intervals,
            est_pitches,
            pitch_tolerance=0,     # no numerical tolerance for midi note numbers
            onset_tolerance=0.05,  # +- 50 ms
            offset_ratio=None,     # do not evaluate offsets
            strict=False
        )
        print('onsets p {:>4.2f} r {:>4.2f} f {:>4.2f} o {:>4.2f}'.format(p, r, f, o))

        # evaluate notes and pitches
        p, r, f, o = mir_eval.transcription.precision_recall_f1_overlap(
            ref_intervals,
            ref_pitches,
            est_intervals,
            est_pitches,
            pitch_tolerance=0,     # no numerical tolerance for midi note numbers
            onset_tolerance=0.05,  # +- 50 ms
            offset_ratio=0.2,      # evaluate complete notes
            strict=False
        )
        print('notes p {:>4.2f} r {:>4.2f} f {:>4.2f} o {:>4.2f}'.format(p, r, f, o))
    else:
        print('complete failure')


def line_segments(intervals, pitches, pitch_offset, yoffset=0.):
    segments = []
    for (start, end), _pitch in zip(intervals, pitches):
        pitch = _pitch + pitch_offset + yoffset
        segments.append(((start, pitch), (end, pitch)))
    return segments


def get_rectangles(intervals,
                   pitches,
                   pitch_height,
                   pitch_offset,
                   pitch_multiplier,
                   yoffset=0.,
                   linewidth=1,
                   color='k',
                   fill=False,
                   alpha=1.):
    rectangles = []
    for (start, end), _pitch in zip(intervals, pitches):
        pitch = ((_pitch + pitch_offset) * pitch_multiplier) + yoffset
        xy = (start, pitch)
        width = end - start
        height = pitch_height
        facecolor = color if fill else None
        rectangles.append(mpatches.Rectangle(
            xy,
            width,
            height,
            edgecolor=color,
            facecolor=facecolor,
            fill=fill,
            linewidth=linewidth,
            alpha=alpha
        ))

    return rectangles


def onsets(intervals, pitches, pitch_offset, yoffset=0.):
    points = []
    for (start, _), _pitch in zip(intervals, pitches):
        pitch = _pitch + pitch_offset + yoffset
        points.append([start, pitch])

    return np.array(points)


def offsets(intervals, pitches, pitch_offset, yoffset=0.):
    points = []
    for (_, end), _pitch in zip(intervals, pitches):
        pitch = _pitch + pitch_offset + yoffset
        points.append([end, pitch])

    return np.array(points)


def run_config(activation_filename):
    results = dict()
    onset_note_prob = 0.9
    offset_prob = 0.1
    threshold = 0.5
    activations_bundle = torch.load(activation_filename)
    activations = activations_bundle['activations']
    midifilename = activations_bundle['metadata']['midi_filename']

    ref_intervals, ref_pitches = get_onsets_and_pitch_labels(midifilename)

    fps = 50
    adsr = ADSRNoteTrackingProcessor(
        onset_prob=onset_note_prob,
        note_prob=onset_note_prob,
        offset_prob=offset_prob,
        attack_length=0.04,
        decay_length=0.04,
        release_length=0.02,
        complete=True,
        onset_threshold=threshold,
        note_threshold=threshold,
        fps=fps,
        pitch_offset=21
    )
    notes, paths = adsr.process(activations, clip=1e-2)
    est_intervals = []
    est_pitches = []
    for onset, pitch, duration in notes:
        est_intervals.append([onset, onset + duration])
        est_pitches.append(pitch)
    est_intervals = np.array(est_intervals)
    est_pitches = np.array(est_pitches)

    # convert timing in both ref_intervals and est_intervals into framecounts
    est_intervals = est_intervals * fps
    ref_intervals = ref_intervals * fps

    # convert intervals and pitches into line segments; subtract 21 from pitch values
    ref_segments = line_segments(ref_intervals, ref_pitches, -21)
    est_segments = line_segments(est_intervals, est_pitches, -21, 0.25)

    ref_onsets = onsets(ref_intervals, ref_pitches, -21)
    est_onsets = onsets(est_intervals, est_pitches, -21, 0.25)

    ref_offsets = offsets(ref_intervals, ref_pitches, -21)
    est_offsets = offsets(est_intervals, est_pitches, -21, 0.25)

    ref_color = 'green'
    est_color = 'gray'

    #############################################################################
    # fig, axes = plt.subplots(nrows=3, sharex=True, sharey=True)
    # ax = axes[0]
    # ax.set_title('onsets')
    # ax.imshow(activations[:, :, 1].T, origin='lower', cmap='gray_r', vmin=0, vmax=1)

    # ax = axes[1]
    # ax.set_title('frames')
    # ax.imshow(activations[:, :, 0].T, origin='lower', cmap='gray_r', vmin=0, vmax=1)

    # ax = axes[2]
    # ax.set_title('offsets')
    # ax.imshow(activations[:, :, 2].T, origin='lower', cmap='gray_r', vmin=0, vmax=1)

    # ax = axes[0]
    # ax.add_collection(LineCollection(ref_segments, colors=[ref_color]))
    # ax.add_collection(LineCollection(est_segments, colors=[est_color]))

    # ax = axes[1]
    # ax.scatter(ref_onsets[:, 0], ref_onsets[:, 1], c=[ref_color])
    # ax.scatter(est_onsets[:, 0], est_onsets[:, 1], c=[est_color])

    # ax = axes[2]
    # ax.scatter(ref_offsets[:, 0], ref_offsets[:, 1], c=[ref_color])
    # ax.scatter(est_offsets[:, 0], est_offsets[:, 1], c=[est_color])

    #############################################################################
    ref_rects = get_rectangles(
        ref_intervals,
        ref_pitches,
        pitch_height=3,
        pitch_offset=-21,
        pitch_multiplier=3,
        yoffset=-0.5,
        linewidth=2,
        color=ref_color,
        fill=False,
        alpha=1.
    )
    est_rects = get_rectangles(
        est_intervals,
        est_pitches,
        pitch_height=2.4,
        pitch_offset=-21,
        pitch_multiplier=3,
        yoffset=-0.2,
        linewidth=1.5,
        color=est_color,
        fill=True,
        alpha=0.5
    )

    cmap = LinearSegmentedColormap.from_list('rwb', ['orange', 'white', 'black'])

    merged = np.zeros((len(activations), 88 * 3))
    merged[:, 0::3] = activations[:, :, 0]  # frames
    merged[:, 1::3] = activations[:, :, 1]  # onsets
    merged[:, 2::3] = activations[:, :, 2]  # offsets
    merged[merged <= 0.5] *= -1

    fig, ax = plt.subplots()
    ax.imshow(merged.T, origin='lower', cmap=cmap, vmin=-1, vmax=1)
    ax.add_collection(PatchCollection(ref_rects, match_original=True))
    ax.add_collection(PatchCollection(est_rects, match_original=True))
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('activation_filename', type=str)

    args = parser.parse_args()
    run_config(args.activation_filename)


if __name__ == '__main__':
    main()
