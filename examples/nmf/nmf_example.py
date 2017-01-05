# Load an audio file, generate a magnitude spectrogram, and factorize using NMF.
#
# Author: Brian K Vogel (brian.vogel@gmail.com)
#

import sys
sys.path =  ['../../src_python/'] + sys.path
import arrayUtils
import os
import wave
import matplotlib.pyplot as plt
import numpy as np


###############################################################
# Name of the C++ function to run:
RUN_CPP_FUNCTION_NAME = 'nmf_example_1'

# Edit to supply the path to a mono wav file.
IN_AUDIO_FILE = '/home/brian/data/wav_files/bkvhi_16khz.wav'

# Adjust the FFT size.
FFT_SIZE = 512

# Adjust the number of 'hop' samples between applications of the FFT.
HOP_SAMPLES = 160
###############################################################


def wav_to_sig(wav_file):
    """Return the signal and sample rate as a tuple.
    This currently only works for a mono audio file.
    """
    spf = wave.open(wav_file,'r')
    sig = spf.readframes(-1)
    sig = np.fromstring(sig, 'Int16')
    fs = spf.getframerate()
    return (sig, fs)


def stft(x, fft_size, hopsamp):
    """Compute the Short-Time Fourier Transform (STFT)
    """
    w = np.hamming(fft_size)
    return np.array([np.fft.rfft(w*x[i:i+fft_size]) 
                     for i in range(0, len(x)-fft_size, hopsamp)])


def make_magnitude_spectrogram():
    (sig, fs) = wav_to_sig(IN_AUDIO_FILE)
    print('Sample rate = {}  Hz'.format(fs))
    spectrogram = stft(sig, FFT_SIZE, HOP_SAMPLES).T
    X = abs(spectrogram)**0.3 # Apply root compression.
    # Normlize so that value are in the range [0,1]
    X = X/X.max()
    arrayUtils.writeArray(X, "X.dat")


if __name__ == '__main__':
    make_magnitude_spectrogram()
    runExe = '../../src_cpp/main ' + RUN_CPP_FUNCTION_NAME
    print(runExe)
    os.system(runExe)

