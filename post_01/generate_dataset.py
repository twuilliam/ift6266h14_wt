
import numpy as np
import cPickle
import glob

# Paths for input/output files
filepath = '/data/timit/readable/'
outputpath = './'

paths = glob.glob(filepath+'TIMIT/TRAIN/DR1/FCJF0/*.npy')
nb_files = len(paths)

# Parameters
Fs = 16000.0
winLength = 241
extr_data_all = np.zeros((0, winLength))


def featExtract(data, winLength):
    """
    Extracts windows of length 'winLength'
    as features from a 1D array of time values
    """
    dataLength = data.shape[0]
    nbWin = dataLength - winLength
    extractData = np.zeros((nbWin, winLength))
    for i in range(nbWin):
        extractData[i, :] = data[i:i+winLength]
    return extractData

i = 1
for files in paths:
    data = np.load(files)
    print(str(i)+'/'+str(nb_files)+': '+files+' loaded...')
    i += 1
    extr_data = featExtract(data, winLength)
    if files == paths[-1]:
        sentence = extr_data/560
    else:
        extr_data_all = np.vstack((extr_data_all, extr_data/560))
    del extr_data

np.random.seed(1234)
np.random.shuffle(extr_data_all)

# Split the data matrix in order to get a vector y with the values to predict
n = len(extr_data_all)
data = np.split(extr_data_all, [winLength - 1], axis=1)
sentence = np.split(sentence, [winLength - 1], axis=1)
del extr_data_all

# Split the data into training, validation, testing sets
train = (data[0][0:int(n/4*2)], data[1][0:int(n/4*2)].reshape((int(n/4*2))))
valid = (data[0][int(n/4*2):int(n/4*3)], data[1][int(n/4*2):int(n/4*3)].reshape(int(n/4)))
test = (data[0][int(n/4*3):], data[1][int(n/4*3):].reshape(int(n-int(n/4*3))))
del data

# Save the extracted features in a .npy file
f = file(outputpath+'timit_train.pkl', 'wb')
cPickle.dump((train, valid, test, sentence), f, protocol=cPickle.HIGHEST_PROTOCOL)
f.close()
