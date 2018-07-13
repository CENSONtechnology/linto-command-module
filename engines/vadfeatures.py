# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 10:12:32 2018

@author: rbaraglia
"""
import numpy as np
import wave
        
def energy(frame):
    """ Returns the  Short Term Energy of a given frame
    input:
        - frame:        a single dimension numerical signal frame
    output:
        - Short Term Energy
    """
    return np.sum(frame**2)/len(frame)

def FBAR(frame, freqs, freq_low, freq_high):
    """ Frequency Band Amplitude Ratio returns the ratio of a given frequency band whitin the frequencies domain.
    input:
        - frame:        a single dimension numerical signal frame
        - freqs:        the frequencies of the frame in the frequency domain (given by spectral_frequencies)
        - freq_low:     the lower limit of the band
        - freq_high:    the upper limit of the band
    output:
        - single value between 0 and 1
    """
    spectrum = np.abs(np.fft.fft(frame))[:int(len(frame)/2)]
    lim_freq = np.searchsorted(freqs, [freq_low, freq_high])
    return np.sum(spectrum[lim_freq[0]: lim_freq[1]])/np.sum(spectrum) if np.sum(spectrum) != 0 else 0


def logenergy(frame):
    en = energy(frame)
    return np.log2(en) if en > 0 else 0
 
def spectral_density(frame):
    """ Returns the spectral density of a given signal frame.
    input:
        - frame:        a single dimension numerical signal frame
    output:
        - spectrum      a frequency vector.
    See also: spectral_frequencies return the matching frequencies for spectrum values. 
    """
    output = np.abs(np.fft.fft(frame))
    return output[1:int(len(frame)/2)]


def spectralFrequencies(frame_length, fs):
    """ Return the frequencies in the spectral domain for a frame.
    input:
        -frame_length: the frame length
        -sampling rate
    output:
        - an array of frequency matching each value in the frequency domain.
    """
    return (fs/2) * np.linspace(0,1,int(frame_length/2))

def split(signal, window_length, overlap=0):
    """ Generator that split a array into frame
    input:
        -signal:        1D array
        -window_length: frame length
        -overlap:       step between each frame
    output:
        - frame:        a frame generator
    """
    signal_length = len(signal)
    i = 0
    while i + window_length <= signal_length:
        yield signal[i:i+window_length]
        i += window_length - overlap


def zcr(frame, norm=True):
    """ Return the zero crossing rate of the given signal frame
    input:
        - frame: A numerical signal frame.
    output:
        - zero crossing rate (scalar)
    """
    return np.sum(np.not_equal(np.sign(frame[1:]), np.sign(frame[:-1]))) /(len(frame) if norm else 1)