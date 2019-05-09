import torch
import argparse
import numpy as np
import mir_eval
from madmom.io import midi
from adsr import ADSRNoteTrackingProcessor
import warnings
import time
import os
np.set_printoptions(precision=4)


def get_onsets_and_pitch_labels(midifile):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        pattern = midi.MIDIFile(midifile)
        intervals = []
        labels = []
        for onset, _pitch, duration, velocity, _channel in pattern.sustained_notes:
            label = int(_pitch)  # do not subtract 21; mir_eval needs pitches strictly >= 0 anyways
            intervals.append([onset, onset + duration])
            labels.append(label)
        return np.array(intervals), np.array(labels)


def run_trial(prediction_filenames, onset_note_prob, offset_prob, threshold):
    prfo_onsets = []
    prfo_notes = []

    t_start = time.time()

    for prediction_filename in prediction_filenames:
        predictions = torch.load(prediction_filename)
        midifilename = predictions['metadata']['midi_filename']
        activations = predictions['activations']

        ref_intervals, ref_pitches = get_onsets_and_pitch_labels(midifilename)
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
            fps=50,
            pitch_offset=21
        )
        notes, paths = adsr.process(activations, clip=1e-2)
        if notes.shape[1] > 0:

            est_intervals = []
            est_pitches = []
            for onset, pitch, duration in notes:
                est_intervals.append([onset, onset + duration])
                est_pitches.append(pitch)
            est_intervals = np.array(est_intervals)
            est_pitches = np.array(est_pitches)

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
            prfo_onsets.append([p, r, f, o])

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
            prfo_notes.append([p, r, f, o])

    ###########################################################
    if len(prfo_notes) > 0:
            result = np.mean(prfo_notes, axis=0)
    else:
        result = [0., 0., 0., 0.]
    return result


def run_config(args, activations_directory):
    prediction_filenames = []
    for entry in sorted(os.listdir(activations_directory)):
        if entry.endswith('.pkl'):
            prediction_filenames.append(os.path.join(activations_directory, entry))

    if args.sample_size < 0 or args.sample_size > len(prediction_filenames):
        print('using all available activations')
    else:
        print('selecting a random sample of size {}'.format(args.sample_size))
        prediction_filenames = np.random.choice(prediction_filenames, args.sample_size)

    results = dict()
    # for onset_note_prob in [0.1, 0.3, 0.5, 0.7, 0.9]:
    #     for offset_prob in [0.1, 0.3, 0.5, 0.7, 0.9]:
    #         for threshold in [0.1, 0.3, 0.5, 0.7, 0.9]:
    for onset_note_prob in [0.9]:
        for offset_prob in [0.1]:
            for threshold in [0.5]:
                t_start = time.time()
                trial = 'onnp_{}_offp_{}_thrs_{}'.format(onset_note_prob, offset_prob, threshold)
                print('begin trial {}'.format(trial))
                results[trial] = run_trial(
                    prediction_filenames,
                    onset_note_prob,
                    offset_prob,
                    threshold
                )
                t_end = time.time()
                print('results[{}]'.format(trial), results[trial])
                print('time in [s]', t_end - t_start)
                # save after each trial ...
                torch.save(results, args.resultfile)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('activations_directory', type=str)
    parser.add_argument('resultfile', type=str)
    parser.add_argument('--sample_size', type=int, default=-1)

    args = parser.parse_args()
    run_config(args, args.activations_directory)


if __name__ == '__main__':
    main()
