#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Provider read an audio stream from default audio input, extract MFCC features and push them into a Queue."""

__author__ = 'Rudy BARAGLIA'
__email__ = 'rbaraglia@linagora.com'
__credits__ = []

import os
import time
from queue import Queue
from threading import Thread

import logging
import configparser
from speechpy.feature import mfcc
import numpy as np

logger = logging.getLogger(__name__)

class Condition:
    """Simple condition to be shared between threads."""
    state = True

class Microphone(Thread):
    def __init__(self, config, raw_queue : Queue, mfcc_queue : Queue, condition: Condition):
        import pyaudio
        Thread.__init__(self)
        self.config = config
        self.raw_queue = raw_queue
        self.mfcc_queue = mfcc_queue
        self.condition = condition
        self.buff_num = []
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(format=pyaudio.paInt16,
                        channels=int(self.config['channel']),
                        rate=int(self.config['sampling_rate']),
                        input=True,
                        frames_per_buffer=int(self.config['chunk_size']))
        self.provide_mfcc = True
        self.provide_raw = False
        self.reactivation_frames = int(self.config['reactivation_delay']) * (int(self.config['sampling_rate']) / int(self.config['chunk_size'])) 
    
    def activate_mfcc_providing(self):
        self.provide_mfcc = True
        self.waiting_frame = self.reactivation_frames

    def run(self):
        logger.info("Started Mic sink audio provider")
        self.waiting_frame = 0 
        while self.condition.state:
            raw_buffer = self.stream.read(int(self.config['chunk_size']), exception_on_overflow=False)
            num_buffer = np.frombuffer(raw_buffer, dtype='<i2').astype(np.float32, order='C')
            self.raw_queue.put(num_buffer)
            if self.provide_mfcc:
                if self.waiting_frame > 0:
                    self.waiting_frame -= 1
                    continue
                self.buff_num = np.concatenate([self.buff_num, num_buffer / 32768.0])
                if len(self.buff_num) >= float(self.config['mfcc_frame_duration']):
                    features = array_to_features(self.buff_num, self.config)
                    self.buff_num = self.buff_num[len(features)*int(self.config['chunk_size']):]
                    self.mfcc_queue.put(features)
        self.stream.close()
        logger.info("Stopped Mic Sink audio provider")

def array_to_features(data: list, config) -> list:
    """Takes a list of normalized audio signal amplitude and returns a list of MFCC parameters"""
    features = mfcc(data, int(config['sampling_rate']),
                    frame_length=float(config['mfcc_frame_duration']),
                    frame_stride=float(config['mfcc_frame_stride']), 
                    num_cepstral=int(config['mfcc_num_cepstral']), 
                    num_filters=int(config['mfcc_num_filters']), 
                    fft_length=int(config['mfcc_fft_length']))
    return features