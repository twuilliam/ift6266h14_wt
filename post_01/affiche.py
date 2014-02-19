
import matplotlib.pyplot as plt
import scipy.io.wavfile as wv


(fs, raw) = wv.read('SX397.WAV')
(fs, pred) = wv.read('predicted_data.wav')

winLength = 241

plt.subplot(311)
plt.plot(raw[winLength-1:pred.size+winLength-1], '-r')
plt.legend(['Raw'])
plt.ylim([-4000, 4000])
plt.subplot(312)
plt.plot(pred, '-b')
plt.legend(['Reconstructed'])
plt.ylim([-4000, 4000])
plt.subplot(313)
plt.plot(pred-raw[winLength-1:pred.size+winLength-1], '-g')
plt.legend(['Difference'])
plt.ylim([-4000, 4000])

plt.show()
