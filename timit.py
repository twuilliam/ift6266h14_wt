"""
Get the indexes for TIMIT dataset for "on the fly"
mini-batch generation in Theano

adapted from:
https://github.com/vdumoulin/research/blob/master/code/pylearn2/datasets/timit.py
"""

import os.path
import numpy as np
from pylearn2.utils import serial


class TIMIT:
    """
    Frame-based TIMIT dataset
    """
    _data_path = '/Users/wthong/Data/'

    # Mean and standard deviation of the acoustic samples from the whole
    # dataset (train, valid, test).
    _mean = 0.0035805809921434142
    _std = 542.48824133746177

    def __init__(self, which_set, frame_length, overlap=0,
                 start=0, stop=None, audio_only=False):
        """
        Parameters
        ----------
        which_set : str
            Either "train", "valid" or "test"
        frame_length : int
            Number of acoustic samples contained in a frame
        overlap : int, optional
            Number of overlapping acoustic samples for two consecutive frames.
            Defaults to 0, meaning frames don't overlap.
        start : int, optional
            Starting index of the sequences to use. Defaults to 0.
        stop : int, optional
            Ending index of the sequences to use. Defaults to `None`, meaning
            sequences are selected all the way to the end of the array.
        audio_only : bool, optional
            Whether to load only the raw audio and no auxiliary information.
            Defaults to `False`.
        """
        self.frame_length = frame_length
        self.overlap = overlap
        self.audio_only = audio_only

        # Load data from disk
        self._load_data(TIMIT._data_path, which_set)
        # Standardize data
        for i, sequence in enumerate(self.raw_wav):
            self.raw_wav[i] = (sequence - TIMIT._mean) / TIMIT._std

        # Slice data
        if stop is not None:
            self.raw_wav = self.raw_wav[start:stop]
            if not self.audio_only:
                self.phones = self.phones[start:stop]
                self.phonemes = self.phonemes[start:stop]
                self.words = self.words[start:stop]
        else:
            self.raw_wav = self.raw_wav[start:]
            if not self.audio_only:
                self.phones = self.phones[start:]
                self.phonemes = self.phonemes[start:]
                self.words = self.words[start:]

        for sequence_id, samples_sequence in enumerate(self.raw_wav):
            test = self.extract_windows_indexes(samples_sequence.size)

    def extract_windows_indexes(self, sample_sequence_size):
        num_frames = sample_sequence_size - self.frame_length

        return num_frames

    def _load_data(self, data_path, which_set):
        """
        Load the TIMIT data from disk.

        Parameters
        ----------
        which_set : str
            Subset of the dataset to use (either "train", "valid" or "test")
        """
        # Check which_set
        if which_set not in ['train', 'valid', 'test']:
            raise ValueError(which_set + " is not a recognized value. " +
                             "Valid values are ['train', 'valid', 'test'].")

        # Create file paths
        timit_base_path = os.path.join(data_path, "timit/readable")
        speaker_info_list_path = os.path.join(timit_base_path, "spkrinfo.npy")
        phonemes_list_path = os.path.join(timit_base_path,
                                          "reduced_phonemes.pkl")
        words_list_path = os.path.join(timit_base_path, "words.pkl")
        speaker_features_list_path = os.path.join(timit_base_path,
                                                  "spkr_feature_names.pkl")
        speaker_id_list_path = os.path.join(timit_base_path,
                                            "speakers_ids.pkl")
        raw_wav_path = os.path.join(timit_base_path, which_set + "_x_raw.npy")
        phonemes_path = os.path.join(timit_base_path,
                                     which_set + "_x_phonemes.npy")
        phones_path = os.path.join(timit_base_path,
                                   which_set + "_x_phones.npy")
        words_path = os.path.join(timit_base_path, which_set + "_x_words.npy")
        speaker_path = os.path.join(timit_base_path,
                                    which_set + "_spkr.npy")

        # Load data. For now most of it is not used, as only the acoustic
        # samples are provided, but this is bound to change eventually.
        # Global data
        if not self.audio_only:
            self.speaker_info_list = serial.load(
                speaker_info_list_path
            ).tolist().toarray()
            self.speaker_id_list = serial.load(speaker_id_list_path)
            self.speaker_features_list = serial.load(speaker_features_list_path)
            self.words_list = serial.load(words_list_path)
            self.phonemes_list = serial.load(phonemes_list_path)
        # Set-related data
        self.raw_wav = serial.load(raw_wav_path)
        if not self.audio_only:
            self.phonemes = serial.load(phonemes_path)
            self.phones = serial.load(phones_path)
            self.words = serial.load(words_path)
            self.speaker_id = np.asarray(serial.load(speaker_path), 'int')


if __name__ == "__main__":
    # train_timit = TIMIT("train", frame_length=240, overlap=10,
    #                     frames_per_example=5)
    valid_timit = TIMIT("valid", frame_length=1, overlap=0, audio_only=False)
    # test_timit = TIMIT("test", frame_length=240, overlap=10,
    #                     frames_per_example=5)
    # import pdb; pdb.set_trace()
#    it = valid_timit.iterator(mode='random_uniform', num_batches=100, batch_size=256)
#    import pdb; pdb.set_trace()
#    for (f, t) in it:
#        print f.shape
