import torch
import torch.nn as nn
import torch.nn.init as init
from torch.distributions.normal import Normal
from sklearn.metrics import precision_recall_fscore_support as prfs
import numpy as np

import mir_eval
from madmom.io import midi
from adsr import ADSRNoteTrackingProcessor
from collections import defaultdict


def get_onsets_and_pitch_labels(midifile):
    pattern = midi.MIDIFile(midifile)
    intervals = []
    labels = []
    for onset, _pitch, duration, velocity, _channel in pattern.sustained_notes:
        label = int(_pitch)  # do not subtract 21; mir_eval needs pitches strictly >= 0 anyways
        intervals.append([onset, onset + duration])
        labels.append(label)
    return np.array(intervals), np.array(labels)


class GaussianDropout(nn.Module):
    def __init__(self, rate):
        super().__init__()
        self.dist = Normal(
            torch.cuda.FloatTensor([1.]),
            torch.cuda.FloatTensor([np.sqrt(rate / (1 - rate))])
        )

    def forward(self, x):
        if self.training:
            noise = self.dist.sample(x.size()).squeeze(-1)
            return x * noise
        else:
            return x


class GaussianNoise(nn.Module):
    def __init__(self, stddev):
        super().__init__()
        self.dist = Normal(
            torch.cuda.FloatTensor([0.]),
            torch.cuda.FloatTensor([stddev])
        )

    def forward(self, x):
        if self.training:
            noise = self.dist.sample(x.size()).squeeze(-1)
            return x + noise
        else:
            return x


class Flatten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.view(x.size()[0], -1)


class Net(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.batch_norm = nn.BatchNorm2d(1)
        self.conv_stem = nn.Sequential(
            nn.Conv2d(1, 30, (3, 3), bias=True),
            nn.ELU(),
            GaussianDropout(0.1),
            GaussianNoise(0.1),

            nn.Conv2d(30, 30, (1, 35), bias=True),
            nn.ELU(),
            GaussianDropout(0.1),
            GaussianNoise(0.1),

            nn.Conv2d(30, 30, (7, 1), bias=True),
            nn.ELU(),
            GaussianDropout(0.1),
            GaussianNoise(0.1)
        )

        self.note_frames = nn.Sequential(
            nn.Conv2d(30, 10, (3, 3), bias=True),
            nn.ELU(),
            GaussianDropout(0.5),
            GaussianNoise(0.1),

            Flatten(),

            nn.Linear(1060, 88)
        )

        self.note_onsets = nn.Sequential(
            nn.Conv2d(30, 10, (3, 3), bias=True),
            nn.ELU(),
            GaussianDropout(0.5),
            GaussianNoise(0.1),

            Flatten(),

            nn.Linear(1060, 88)
        )

        self.note_offsets = nn.Sequential(
            nn.Conv2d(30, 10, (3, 3), bias=True),
            nn.ELU(),
            GaussianDropout(0.5),
            GaussianNoise(0.1),

            Flatten(),

            nn.Linear(1060, 88)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.best = dict(
            f=0,
            loss=10.
        )

    def forward(self, batch):
        h_bn = self.batch_norm(batch['x'])
        h_stem = self.conv_stem(h_bn)

        y_onsets = self.note_onsets(h_stem)
        y_frames = self.note_frames(h_stem)
        y_offsets = self.note_offsets(h_stem)

        if self.training:
            return dict(
                y_onsets=y_onsets,
                y_frames=y_frames,
                y_offsets=y_offsets
            )
        else:
            return dict(
                y_onsets=torch.sigmoid(y_onsets),
                y_frames=torch.sigmoid(y_frames),
                y_offsets=torch.sigmoid(y_offsets)
            )

    def get_train_loss_function(self):
        lambdas = self.config['lambdas']
        bce_onsets = nn.BCEWithLogitsLoss(reduction='mean')
        bce_frames = nn.BCEWithLogitsLoss(reduction='mean')
        bce_offsets = nn.BCEWithLogitsLoss(reduction='mean')

        def loss_function(output, target):
            loss = lambdas['y_onsets'] * bce_onsets(output['y_onsets'], target['y_onsets'])
            loss += lambdas['y_frames'] * bce_frames(output['y_frames'], target['y_frames'])
            loss += lambdas['y_offsets'] * bce_offsets(output['y_offsets'], target['y_offsets'])
            return loss

        return loss_function

    def evaluate_adsr(self, metadata, predictions):
        clip = 1e-2
        # see adsr.py for this!
        activations = np.stack([
            predictions['y_frames'],
            predictions['y_onsets'],
            predictions['y_offsets']
        ], axis=-1)

        # import matplotlib.pyplot as plt
        # fig, axes = plt.subplots(nrows=3, sharex=True, sharey=True)
        # axes[0].imshow(activations[:, :, 0].T, origin='lower')
        # axes[1].imshow(activations[:, :, 1].T, origin='lower')
        # axes[2].imshow(activations[:, :, 2].T, origin='lower')
        # plt.show()

        midifilename = metadata['midi_filename']
        ref_intervals, ref_pitches = get_onsets_and_pitch_labels(midifilename)

        results = dict()
        # this is just to get an approximate feeling for the whole note performance
        # currently. we'll tune this after training with a gridsearch
        oothresholds = [
            [0.8, 0.1, 0.4],
        ]

        for onset_note_prob, offset_prob, threshold in oothresholds:
            trial = 'onnp_{}_offp_{}_thrs_{}'.format(
                onset_note_prob,
                offset_prob,
                threshold
            )

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
            notes, __paths = adsr.process(activations, clip=1e-2)

            if notes.shape[1] > 0:
                est_intervals = []
                est_pitches = []
                for onset, pitch, duration in notes:
                    est_intervals.append([onset, onset + duration])
                    est_pitches.append(pitch)
                est_intervals = np.array(est_intervals)
                est_pitches = np.array(est_pitches)

                # evaluate onsets and pitches
                on_p, on_r, on_f, on_o = mir_eval.transcription.precision_recall_f1_overlap(
                    ref_intervals,
                    ref_pitches,
                    est_intervals,
                    est_pitches,
                    pitch_tolerance=0,     # no numerical tolerance for midi note numbers
                    onset_tolerance=0.05,  # +- 50 ms
                    offset_ratio=None,     # do not evaluate offsets
                    strict=False
                )

                # evaluate notes and pitches
                fu_p, fu_r, fu_f, fu_o = mir_eval.transcription.precision_recall_f1_overlap(
                    ref_intervals,
                    ref_pitches,
                    est_intervals,
                    est_pitches,
                    pitch_tolerance=0,     # no numerical tolerance for midi note numbers
                    onset_tolerance=0.05,  # +- 50 ms
                    offset_ratio=0.2,      # evaluate complete notes
                    strict=False
                )
                results[trial] = dict(
                    onsets=dict(p=on_p, r=on_r, f=on_f, o=on_o),
                    full=dict(p=fu_p, r=fu_r, f=fu_f, o=fu_o)
                )
            else:
                results[trial] = dict(
                    onsets=dict(p=0, r=0, f=0, o=0),
                    full=dict(p=0, r=0, f=0, o=0)
                )
        return results

    def evalute_one(self, metadata, predictions, batches):
        def log_loss(y, _p):
            eps = 1e-3
            p = np.clip(_p, eps, 1. - eps)
            return np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p)))

        outputs = ['y_onsets', 'y_frames', 'y_offsets']

        result = dict()
        for output in outputs:
            loss = log_loss(batches[output], predictions[output])
            y_true = (batches[output] > 0.5) * 1
            y_pred = (predictions[output] > 0.5) * 1

            p, r, f, _ = prfs(y_true, y_pred, average='micro')

            result[output] = dict(
                loss=loss,
                p=p,
                r=r,
                f=f
            )

        result['adsr'] = self.evaluate_adsr(metadata, predictions)

        return result

    def evaluate_aggregate_checkpoint(self,
                                      name,
                                      all_predictions,
                                      all_batches,
                                      logger,
                                      epoch,
                                      scheduler):
        results = []
        for ip, ib in zip(all_predictions, all_batches):
            results.append(self.evalute_one(
                ip['metadata'],
                ip['predictions'],
                ib['batches']
            ))

        outputs = ['y_onsets', 'y_frames', 'y_offsets']

        mean_loss = 0
        mean_p = 0
        mean_r = 0
        mean_f = 0
        for output in outputs:
            loss, p, r, f = 0, 0, 0, 0
            for result in results:
                loss += result[output]['loss']
                p += result[output]['p']
                r += result[output]['r']
                f += result[output]['f']
            loss /= len(results)
            p /= len(results)
            r /= len(results)
            f /= len(results)

            mean_loss += loss
            mean_p += p
            mean_r += r
            mean_f += f

            logger.add_scalar('{}_individual_losses/{}_loss'.format(name, output), loss, global_step=epoch)
            logger.add_scalar('{}_individual_p/{}_p'.format(name, output), p, global_step=epoch)
            logger.add_scalar('{}_individual_r/{}_r'.format(name, output), r, global_step=epoch)
            logger.add_scalar('{}_individual_f/{}_f'.format(name, output), f, global_step=epoch)

        mean_loss /= len(outputs)
        mean_f /= len(outputs)
        mean_p /= len(outputs)
        mean_r /= len(outputs)

        logger.add_scalar('{}_means/loss'.format(name), mean_loss, global_step=epoch)
        logger.add_scalar('{}_means/p'.format(name), mean_p, global_step=epoch)
        logger.add_scalar('{}_means/r'.format(name), mean_r, global_step=epoch)
        logger.add_scalar('{}_means/f'.format(name), mean_f, global_step=epoch)

        #####################################################################
        # adsr eval
        trials = defaultdict(list)
        for result in results:
            for trial_key, trial_result in result['adsr'].items():
                for what in ['onsets', 'full']:
                    for prfo in ['p', 'r', 'f', 'o']:
                        flat_trial_key = '{}_{}_{}/{}'.format(name, what, trial_key, prfo)
                        trials[flat_trial_key].append(
                            trial_result[what][prfo]
                        )
        for flat_trial_key in trials.keys():
            trials[flat_trial_key] = np.mean(trials[flat_trial_key])

        for flat_trial_key, flat_trial_result in trials.items():
            logger.add_scalar(flat_trial_key, flat_trial_result, global_step=epoch)

        checkpoint_filename = None
        if name == 'valid':
            if mean_f >= self.best['f']:
                self.best['f'] = mean_f

            if mean_loss <= self.best['loss']:
                checkpoint_filename = 'runs/{}/best_valid_loss.pkl'.format(
                    self.config['run_id']
                )
                self.best['loss'] = mean_loss

        logger.add_scalar('{}_best/f'.format(name), self.best['f'], global_step=epoch)
        logger.add_scalar('{}_best/loss'.format(name), self.best['loss'], global_step=epoch)
        return checkpoint_filename
