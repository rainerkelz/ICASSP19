import numpy as np
import madmom
from madmom.processors import Processor
from madmom.ml.hmm import TransitionModel, ObservationModel, HiddenMarkovModel


class ADSRStateSpace(object):
    """
    gives the states a meaning

    """

    def __init__(self, attack_length=1, decay_length=1, release_length=1):
        # define note with states which must be transitioned
        self.silence = 0
        self.attack = 1
        self.decay = self.attack + attack_length
        self.sustain = self.decay + decay_length
        self.release = self.sustain + release_length

    @property
    def num_states(self):
        return self.release + 1


# transition models
class ADSRTransitionModel(TransitionModel):
    """
    Transition model for note transcription with a HMM.

    Parameters
    ----------
    state_space : :class:`ADSRStateSpace` instance
        NoteStateSpace instance.

    """

    def __init__(self, state_space, onset_prob=0.5, note_prob=0.5, offset_prob=0.5, end_prob=1.):
        # save attributes
        self.state_space = state_space
        # states
        silence = state_space.silence
        attack = state_space.attack
        decay = state_space.decay
        sustain = state_space.sustain
        release = state_space.release
        # transitions = [(from_state, to_state, prob), ...]
        # onset phase & min_onset_length
        t = [(silence, silence, 1. - onset_prob),
             (silence, attack, onset_prob)]
        for s in range(attack, decay):
            t.append((s, silence, 1. - onset_prob))
            t.append((s, s + 1, onset_prob))
        # transition to note & min_note_duration
        for s in range(decay, sustain):
            t.append((s, silence, 1. - note_prob))
            t.append((s, s + 1, note_prob))
        # 3 possibilities to continue note
        prob_sum = onset_prob + note_prob + offset_prob
        # 1) sustain note (keep sounding)
        t.append((sustain, sustain, note_prob / prob_sum))
        # 2) new note
        t.append((sustain, attack, onset_prob / prob_sum))
        # 3) release note (end note)
        t.append((sustain, sustain + 1, offset_prob / prob_sum))
        # release phase
        for s in range(sustain + 1, release):
            t.append((s, sustain, offset_prob))
            t.append((s, s + 1, 1. - offset_prob))
        # after releasing a note, go back to silence or start new note
        t.append((release, silence, end_prob))
        t.append((release, release, 1. - end_prob))
        t = np.array(t)
#         print(t)
        # make the transitions sparse
        t = self.make_sparse(t[:, 1].astype(np.int), t[:, 0].astype(np.int),
                             t[:, 2])
        # instantiate a TransitionModel
        super(ADSRTransitionModel, self).__init__(*t)


class ADSRObservationModel(ObservationModel):
    """
    Observation model for note transcription tracking with a HMM.

    Parameters
    ----------
    state_space : :class:`NoteStateSpace` instance
        NoteStateSpace instance.

    """

    def __init__(self, state_space):
        # define observation pointers
        pointers = np.zeros(state_space.num_states, dtype=np.uint32)
        # map from densities to states
        pointers[state_space.silence:] = 0
        pointers[state_space.attack:] = 1
        pointers[state_space.decay:] = 2
        # sustain uses the same observations as decay
        pointers[state_space.release:] = 3
        # instantiate a ObservationModel with the pointers
        super(ADSRObservationModel, self).__init__(pointers)

    def log_densities(self, observations):
        """
        Computes the log densities of the observations.

        Parameters
        ----------
        observations : tuple with two numpy arrays
            Observations (i.e. 3d activations of the CNN).

        Returns
        -------
        numpy array
            Log densities of the observations.

        """
        # observations: notes, onsets, offsets
        densities = np.ones((len(observations), 4), dtype=np.float)
        # silence
        # densities[:, 0] = 1. - np.sum(observations, axis=1)
        densities[:, 0] = 1. - observations[:, 1]
        # attack
        densities[:, 1] = observations[:, 1]
        # decay
        densities[:, 2] = observations[:, 0]
        # release
        densities[:, 3] = observations[:, 2]
        # return the log densities
        return np.log(densities)


class ADSRNoteTrackingProcessor(Processor):

    ONSET_PROB = 0.8
    NOTE_PROB = 0.8
    OFFSET_PROB = 0.15

    pitch_offset = 21

    def __init__(self, onset_prob=0.8, note_prob=0.8, offset_prob=0.15, end_prob=0.8,
                 attack_length=0.04, decay_length=0.04, release_length=0.02,
                 onset_threshold=None, note_threshold=None, complete=True, fps=50., **kwargs):
        # state space
        self.st = ADSRStateSpace(attack_length=int(attack_length * fps),
                                 decay_length=int(decay_length * fps),
                                 release_length=int(release_length * fps))
        # transition model
        self.tm = ADSRTransitionModel(self.st, onset_prob=onset_prob, note_prob=note_prob,
                                      offset_prob=offset_prob, end_prob=end_prob)
        # observation model
        self.om = ADSRObservationModel(self.st)
        # instantiate a HMM
        self.hmm = HiddenMarkovModel(self.tm, self.om, None)
        # save variables
        self.onset_threshold = onset_threshold
        self.note_threshold = note_threshold
        self.complete = complete
        self.fps = fps

    def process(self, activations, **kwargs):
        """
        Detect the notes in the given activation function.

        Parameters
        ----------
        activations : numpy array
            Note activation function.

        Returns
        -------
        onsets : numpy array
            Detected notes [seconds, pitches].

        """
        notes = []
        paths = []
        note_path = np.arange(self.st.attack, self.st.release)
        # process ech pitch individually
        for pitch in range(activations.shape[1]):
            # decode activations for this pitch with HMM
            path, _ = self.hmm.viterbi(activations[:, pitch, :])
            paths.append(path)
            # extract HMM note segments
            segments = np.logical_and(path > self.st.attack,
                                      path < self.st.release)
            # extract start and end positions (transition points)
            idx = np.nonzero(np.diff(segments.astype(np.int)))[0]
            # add end if needed
            if len(idx) % 2 != 0:
                idx = np.append(idx, [len(activations)])
            # all sounding frames
            frames = activations[:, pitch, 0]
            # all frames with onset activations
            onsets = activations[:, pitch, 1]
            # iterate over all segments to decide which to keep
            for onset, offset in idx.reshape((-1, 2)):
                # extract note segment
                segment = path[onset:offset]
                # discard segment which do not contain the complete note path
                if self.complete and np.setdiff1d(note_path, segment).any():
                    continue
                # discard segments without a real note
                if frames[onset:offset].max() < self.note_threshold:
                    continue
                # discard segments without a real onset
                if onsets[onset:offset].max() < self.onset_threshold:
                    continue
                # append segment as note
                notes.append([onset / self.fps, pitch + self.pitch_offset,
                              (offset - onset) / self.fps])

        # sort the notes, convert timing information and return them
        notes = np.array(sorted(notes), ndmin=2)
        return notes, np.array(paths)


def track_adsr(act_file, fps=50., clip=1e-2):
    act = madmom.features.Activations.load(act_file)
    if clip is not None:
        act = np.clip(act, clip, 1. - clip)
    hmm = ADSRNoteTrackingProcessor(onset_prob=0.8, note_prob=0.8, offset_prob=0.2, end_prob=1.,
                                    attack_length=0.04, decay_length=0.04, release_length=0.02,
                                    complete=True, onset_threshold=0.5, note_threshold=0.5, fps=fps)
    # remove all onsets below threshold
    det, hmm_path = hmm(act)
    det_file = act_file[:-3] + 'mid'
    madmom.io.midi.write_midi(det, det_file)
