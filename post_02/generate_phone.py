
import numpy
import os
from pylearn2.utils import serial
import matplotlib.pylab as plot
import scipy.io.wavfile as wv
import time
import sys
from operator import and_

#############
# Load data #
#############

timit_base_path = '/Users/wthong/Data/timit/readable/'
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
speaker_info_list = serial.load(speaker_info_list_path).tolist().toarray()
speaker_id_list = serial.load(speaker_id_list_path)
phonemes_list = serial.load(phonemes_list_path)

# Load data
raw_wav = serial.load(raw_wav_path)
phonemes = serial.load(phonemes_path)
speaker_id = numpy.asarray(serial.load(speaker_path), 'int')

##############
# Parameters #
##############

Fs = 16000.0
winLength = 241
extr_data_all = numpy.zeros((0, winLength))
mean_timit = 0.0035805809921434142
std_timit = 542.48824133746177

outputpath = './'
desired_phn = 'aw'
addnoise = False
index_phn = phonemes_list.index(desired_phn)
dialect = 1
sex = 25  # male=25 | female=24

#################
# Extract phone #
#################

# Search people from the 1st dialect and male
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

total = []

for i, raw in enumerate(raw_wav[indexes]):
    # exclude phonemes that aren't 'aa'
    raw[numpy.where(phonemes[indexes[0][i]] != index_phn)] = 0
    raw = numpy.trim_zeros(raw)
    raw = numpy.split(raw, numpy.where(raw == 0)[0])
    for seq in raw:
        if seq.size > winLength and numpy.compress(seq != 0, seq).size > winLength:
            total.append(numpy.compress(seq != 0, seq))

del raw_wav, phonemes, speaker_id

##################
# Create windows #
##################

def featExtract(data, winLength):
    """
    Extracts windows of length 'winLength'
    as features from a 1D array of time values
    """
    dataLength = data.shape[0]
    nbWin = dataLength - winLength
    extractData = numpy.zeros((nbWin, winLength))
    for i in range(nbWin):
        extractData[i, :] = data[i:i+winLength]
    return extractData

for raw in total:
    extr_data = featExtract(raw, winLength)
    extr_data_all = numpy.vstack((extr_data_all, (extr_data-mean_timit)/std_timit))
    if addnoise:
        extr_data_all = numpy.vstack((extr_data_all, ((extr_data-mean_timit)/std_timit)+[numpy.random.normal(0, 0.15) for i in xrange(winLength)]))
        extr_data_all = numpy.vstack((extr_data_all, ((extr_data-mean_timit)/std_timit)+[numpy.random.normal(0, 0.10) for i in xrange(winLength)]))
        extr_data_all = numpy.vstack((extr_data_all, ((extr_data-mean_timit)/std_timit)+[numpy.random.normal(0, 0.05) for i in xrange(winLength)]))
    del extr_data

##################
# Create dataset #
##################

numpy.random.seed(1234)
numpy.random.shuffle(extr_data_all)

# Split the data matrix in order to get a vector y with the values to predict
n = len(extr_data_all)
data = numpy.split(extr_data_all, [winLength - 1], axis=1)
del extr_data_all

# Split the data into training, validation, testing sets
train = (data[0][0:int(n/4*2)], data[1][0:int(n/4*2)].reshape((int(n/4*2))))
valid = (data[0][int(n/4*2):int(n/4*3)], data[1][int(n/4*2):int(n/4*3)].reshape(int(n/4)))
test = (data[0][int(n/4*3):], data[1][int(n/4*3):].reshape(int(n-int(n/4*3))))
sentence = (numpy.zeros(240), 0.0)
del data

# Save the extracted features in a .npz file
f = file(outputpath+'timit_'+desired_phn+'_train_aug.npz', 'wb')
numpy.savez(f, train=train, valid=valid, test=test, sentence=sentence)
f.close()

wv.write(desired_phn+'.wav', 16000, numpy.hstack(total))
#plot.figure()
#plot.plot(numpy.hstack(total))

