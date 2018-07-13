#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" WUWEngine is the processing unit for Wake-Up-Word spotting, it received audio features through a Queue
and determines if the keyword has been spoken using a preset tensorflow model. 
"""
__author__ = 'Rudy BARAGLIA'
__email__ = 'rbaraglia@linagora.com'
__credits__ = []

import os
from importlib import import_module
from threading import Thread

import logging
import numpy as np
from queue import Queue
import configparser

logger = logging.getLogger(__name__)
FILE_PATH = os.path.dirname(os.path.abspath(__file__)) + '/'

class WUWEngine(Thread):
    def __init__(self, config, feat_queue: Queue, condition, on_spotting: callable):
        """Initialises tensorflow graph"""
        Thread.__init__(self)
        self.config = config

        self.sensitivity = float(self.config['sensitivity']) # Tolerance on detection
        self.triggered = 0 # Number of consecutive positive windows
        self.response_trigger = int(self.config['response_trigger']) # Number of consecutive positive windows needed to trigger detection
        
        self.condition = condition # Condition to maintain the thread alive
        self.feat_queue = feat_queue # Features input
        self.features = np.zeros((int(self.config['n_features']),int(self.config['feature_size']))) # Features buffer
        self.on_spotting = on_spotting # Callback function

        # Loading graph
        self.tf = import_module('tensorflow')
        self.graph = self.load_graph(self.config['model_path'])
        
        self.inp_var = self.graph.get_operation_by_name('import/net_input').outputs[0]
        self.out_var = self.graph.get_operation_by_name('import/net_output').outputs[0]

        self.sess = self.tf.Session(graph=self.graph)
        logger.debug("WUW engine is set")

    def load_graph(self, model_path: str) -> 'tf.graph':
        graph = self.tf.Graph()
        graph_def = self.tf.GraphDef()
        with open(FILE_PATH + model_path, "rb") as f:
            graph_def.ParseFromString(f.read())
        with graph.as_default():
            self.tf.import_graph_def(graph_def)
        return graph

    def run(self):
        # Run until self.condition is set to false
        trigger_values = []
        logger.debug("WUW engine is running")
        while self.condition.state:
            feats = self.feat_queue.get()
            if feats is None:
                continue
            self.features = np.concatenate([self.features[len(feats):], feats])
            res = self.sess.run(self.out_var, {self.inp_var : self.features[np.newaxis]})[0][0]
            # Trigger decision
            if res >= 1 - self.sensitivity:
                self.triggered += 1
                trigger_values.append(res)
                if self.triggered == self.response_trigger:
                    self.on_spotting("wuw-spotted", np.mean(trigger_values))
            else:
                self.triggered = 0
                trigger_values = []


