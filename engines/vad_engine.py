import os
import time
from importlib import import_module
from threading import Thread

import logging
import numpy as np
from collections import deque
from queue import Queue
import configparser

import engines.vadfeatures as vad

logger = logging.getLogger(__name__)

class VADEngine(Thread):
    def __init__(self, config, feat_queue: Queue, condition, on_spotting: callable):
        Thread.__init__(self)
        self.config = config
        self.condition = condition
        self.feat_queue = feat_queue
        self.on_spotting = on_spotting
        self.audio_buffer = np.array([])
        self.energy_th = 20
        self.speech_energy_log = deque([20],maxlen=50)
        self.silence_energy_log = deque([0],maxlen=50)
        self.freqs = vad.spectralFrequencies(int(self.config['window_width']), int(self.config['sampling_rate']))
        self.detecting = False
        logger.debug("VAD engine is set")

    def start_detecting(self):
        logger.debug("VAD in on")
        self.on_spotting("vad_start")
        self.detecting = True
        self.start_time = time.time()
        self.consecutive_silence = 0
        self.speech_window = 0
        self.speech = False
    
    def stop_detecting(self):
        if self.detecting:
            self.on_spotting('canceled')
            self.detecting = False
            logger.debug("VAD is off")

    def run(self):
        logger.debug("VAD engine is running")
        self.start_time = 0
        self.consecutive_silence = 0
        silence_threshold = int(self.config['silence_threshold'])
        speech_threshold = int(self.config['speech_threshold'])
        timeout =  int(self.config['timeout'])
        while self.condition.state:
            #Timeout
            if self.detecting and time.time() > self.start_time + timeout:
                self.detecting = False
                self.on_spotting('timeout')
            new_buff = self.feat_queue.get()
            if new_buff is None:
                break
            self.audio_buffer = np.concatenate([self.audio_buffer,new_buff])
            if len(self.audio_buffer) < int(self.config['window_width']):
                continue
            for window in vad.split(self.audio_buffer,  int(self.config['window_width']),  int(self.config['window_overlap'])):
                en_window = vad.logenergy(window)
                fbar = vad.FBAR(window, self.freqs,  int(self.config['fbar_lfreq']),  int(self.config['fbar_hfreq']))
                #Is speech
                if en_window > self.energy_th and fbar >  float(self.config['fbar_th']):
                    self.speech_energy_log.append(en_window)
                    if self.detecting:
                        self.speech_window += 1
                        print("{}/{}".format(self.speech_window, speech_threshold))
                        self.speech = self.speech_window >= speech_threshold
                    self.consecutive_silence = 0
                    self.start_time = time.time()
                else:
                    self.silence_energy_log.append(en_window)
                    self.consecutive_silence += 1
                    if self.detecting and self.speech and self.consecutive_silence > silence_threshold:
                        self.detecting = False
                        self.on_spotting('thresholdReached')
                        
            self.audio_buffer = self.audio_buffer[len(self.audio_buffer) - len(self.audio_buffer) % int(self.config['window_width']):]

            #update energy threshold
            self.energy_th = np.mean([np.mean(self.speech_energy_log), np.mean(self.silence_energy_log)])
        logger.info("VAD engine is off")
            

            





