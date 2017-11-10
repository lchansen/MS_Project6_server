import scipy.io.wavfile
from audioutility import AudioUtility
import numpy as np

sample_rate, signal = scipy.io.wavfile.read('data/OSR_us_000_0010_8k.wav')
au = AudioUtility(signal=signal,sample_rate=sample_rate)
filter, mfcc = au.get_filter_mfcc()
print(mfcc.shape)
print(mfcc[0])
print("-------")
print(mfcc.flatten())