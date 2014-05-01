
import numpy
import os
import scipy.io.wavfile as wv
from operator import and_
from scipy.stats import mode
import cPickle

#############
# Load data #
#############

timit_base_path = '/datasets/timit/readable/'
which_set = 'train'

# Load paths
speaker_info_list_path = os.path.join(timit_base_path, "spkrinfo.npy")
phonemes_list_path = os.path.join(timit_base_path,
                                  "reduced_phonemes.pkl")
speaker_id_list_path = os.path.join(timit_base_path,
                                    "speakers_ids.pkl")

raw_wav_path = os.path.join(timit_base_path, which_set + "_x_raw.npy")
phonemes_path = os.path.join(timit_base_path,
                             which_set + "_x_phonemes.npy")
speaker_path = os.path.join(timit_base_path,
                            which_set + "_spkr.npy")

# Load info
speaker_info_list = numpy.load(speaker_info_list_path).tolist().toarray()
with open(speaker_id_list_path, 'rb') as f:
    speaker_id_list = cPickle.load(f)
f.close()
with open(phonemes_list_path) as f:
    phonemes_list = cPickle.load(f)
f.close()

# Load data
raw_wav = numpy.load(raw_wav_path)
phonemes = numpy.load(phonemes_path)
speaker_id = numpy.asarray(numpy.load(speaker_path), 'int')

##############
# Parameters #
##############

Fs = 16000.0
winLength = 241
mean_timit = 0.0035805809921434142
std_timit = 542.48824133746177

outputpath = './'
dialect = 1
sex = 25  # male=25 | female=24

###################
# Extract dataset #
###################

print 'Loading data'

# Search people from the 1st dialect and male in the whole dataset
firstdialect = numpy.where(and_(speaker_info_list[:, dialect] == 1,
                                speaker_info_list[:, sex] == 1))[0]

# Select only the data of the training set
firstdialect_train = [firstdialect[dialect] for dialect in range(firstdialect.size)
                      if firstdialect[dialect] in numpy.unique(speaker_id)]

# Create conditional vector (binary)
cond = []
for spkr in speaker_id:
    if spkr in firstdialect_train:
        cond.append(True)
    else:
        cond.append(False)

# Look for indexes of people from the 1st dialect and male
indexes = numpy.where(numpy.array(cond))

total_wav = raw_wav[indexes]
total_phn = phonemes[indexes]

num_phn =  numpy.max([numpy.max(sequence) for sequence in phonemes]) + 1

del raw_wav, phonemes, speaker_id

##################
# Create windows #
##################

class create_whole_dataset(object):
    def __init__(self, num_phn, mean_timit, std_timit, winLength, gap):
        self.num_phn = num_phn
        self.mean = mean_timit
        self.std = std_timit
        self.winL = winLength
        self.gap = int(gap)

    def onehot(self,y):
        z = numpy.zeros(self.num_phn)
        z[y] = 1
        return z

    def featExtract(self, raw, phn, phn_prev, phn_next):
        dataLength = raw.shape[0] - self.winL
        extract_wav = numpy.zeros((0, self.winL))
        extract_phn = numpy.zeros((0, self.num_phn*3), dtype=numpy.int16)
        for i in range(dataLength)[0::self.gap]:
            extract_wav = numpy.vstack((extract_wav, raw[i:i+self.winL]))

            cur_phn = int(phn[i])
            extract_phn = numpy.vstack((extract_phn,
                                        numpy.concatenate((self.onehot(phn_prev),
                                                           self.onehot(cur_phn),
                                                           self.onehot(phn_next)))))
        return extract_wav, extract_phn

    def getSequence(self, chng, whole_raw, whole_phn):
        extr_whole_wav = numpy.zeros((0, self.winL))
        extr_whole_phn = numpy.zeros((0, self.num_phn*3), dtype=numpy.int16)

        for i, j, k in zip(chng, chng[1:], numpy.sort(numpy.append(chng, [0, len(whole_raw)]))):
            # k previous phoneme
            # i current phoneme
            # j next phoneme
            tmp_wav, tmp_phn = self.featExtract(whole_raw[i:j+self.winL], whole_phn[i:j+self.winL], phn[k], phn[j])
            extr_whole_wav = numpy.vstack((extr_whole_wav, tmp_wav))
            extr_whole_phn = numpy.vstack((extr_whole_phn, tmp_phn))
        return (extr_whole_wav-self.mean)/self.std, extr_whole_phn


##################
# Create dataset #
##################

print 'Create dataset'

data_wav_phn = create_whole_dataset(num_phn, mean_timit, std_timit, winLength, 10)

extr_data_all = numpy.zeros((0, winLength))
extr_data = numpy.zeros((0, winLength))
extr_onehot_all = numpy.zeros((0, num_phn*3), dtype=numpy.int16)
extr_onehot = numpy.zeros((0, num_phn*3), dtype=numpy.int16)

for raw, phn in zip(total_wav, total_phn):
    chng_phn = numpy.where(abs(numpy.diff(phn)) > 0)[0] + 1
    extr_data, extr_onehot = data_wav_phn.getSequence(chng_phn, raw, phn)

    extr_data_all = numpy.vstack((extr_data_all, extr_data))
    extr_onehot_all = numpy.vstack((extr_onehot_all, extr_onehot))

del extr_data, extr_onehot, data_wav_phn

print 'Saving dataset'

numpy.random.seed(1234)
n = len(extr_data_all)
random_list = numpy.arange(n)
numpy.random.shuffle(random_list)
train_ind = random_list[0:int(n/4*2)]
valid_ind = random_list[int(n/4*2):int(n/4*3)]
test_ind = random_list[int(n/4*3):]

# Acoustic samples
data = numpy.split(extr_data_all, [winLength - 1], axis=1)

train_wav = (data[0][train_ind], numpy.ndarray.flatten(data[1][train_ind]))
valid_wav = (data[0][valid_ind], numpy.ndarray.flatten(data[1][valid_ind]))
test_wav = (data[0][test_ind], numpy.ndarray.flatten(data[1][test_ind]))
del data, extr_data_all

f = file(outputpath+'timit_acoustic.npz', 'wb')
numpy.savez(f, train=train_wav, valid=valid_wav, test=test_wav)
f.close()

# Phonemes
train_phn = extr_onehot_all[train_ind]
valid_phn = extr_onehot_all[valid_ind]
test_phn = extr_onehot_all[test_ind]
del extr_onehot_all

# Save the extracted features in a .npz file
f = file(outputpath+'timit_phonemes.npz', 'wb')
numpy.savez(f, train=train_phn, valid=valid_phn, test=test_phn)
f.close()

###################
# Create sentence #
###################

# Sentence
extr_onehot_all = numpy.zeros((0, num_phn*3), dtype=numpy.int16)
extr_onehot = numpy.zeros((0, num_phn*3), dtype=numpy.int16)
sentence = create_whole_dataset(num_phn, mean_timit, std_timit, winLength-1, 1)
for raw, phn in zip(total_wav[0:1], total_phn[0:1]):
    print 'Extract frames for the sentence'
    wv.write(os.path.join(outputpath, 'original_data.wav'), 16000, raw)
    chng_phn = numpy.where(abs(numpy.diff(phn)) > 0)[0] + 1
    _, extr_onehot = sentence.getSequence(chng_phn, raw, phn)
    extr_onehot_all = numpy.vstack((extr_onehot_all, extr_onehot))
del sentence, extr_onehot
# Save the extracted features in a .npz file
f = file(outputpath+'timit_sentence.npz', 'wb')
numpy.savez(f, sentence=extr_onehot_all)
f.close()
