#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = 'Rudy BARAGLIA'
__email__ = 'rbaraglia@linagora.com'
__credits__ = []

import os
import sys
import time
import threading
import datetime

import argparse
import json
import configparser
import logging
from queue import Queue
import paho.mqtt.client as mqtt
import tenacity

from engines.wuw_engine import WUWEngine
from engines.vad_engine import VADEngine
from provider import Condition, Microphone

FILE_PATH = os.path.dirname(os.path.abspath(__file__)) + '/'
class VoiceCommander:
    def __init__(self, conf: configparser.ConfigParser):
        """Initialize audio provider, engines and MQTT message handling.


        Keyword arguments:
        args -- Executable parameter
        conf -- Configuration file parameters
        """
        self.config = conf['BROKER']

        #Event binding manifest
        self.event_manifest = json.load(open(FILE_PATH + 'event_binding.json', 'r'))

        #Thread communication
        self.queue_raw = Queue() #Transmits raw audio
        self.queue_mfcc = Queue() # Transmit mfcc features
        self.condition = Condition() #Boolean Object to safely stop thread.

        #Audio provider
        self.provider_thread = Microphone(conf["PROVIDER"], self.queue_raw, self.queue_mfcc, self.condition)

        #WuW engine thread 
        self.wuwengine_thread = WUWEngine(conf["WUW_ENGINE"], self.queue_mfcc, self.condition, on_spotting=self._on_event)

        #VAD engine thread
        self.vadengine_thread = VADEngine(conf["VAD_ENGINE"], self.queue_raw, self.condition, self._on_event)

        #MQTT broker client
        self.broker = self._broker_connect()
        self.broker.on_message = self._on_broker_message
            
    def run(self):
        """Run the wuw spotter and its threads until broker is disconnected."""
        self.provider_thread.start()
        self.wuwengine_thread.start()
        self.vadengine_thread.start()
        try:
            self.broker.loop_forever()
            self.condition.state = False
        except KeyboardInterrupt:
            logging.info("Process interrupted by user")
        finally:
            self.condition.state = False
            self.queue_raw.put(None)
            self.queue_mfcc.put(None)
            self.vadengine_thread.join()
            self.wuwengine_thread.join()
        logging.info("WuW spotter is off.")

    @tenacity.retry(wait=tenacity.wait_random(min=1, max=10),
                retry=tenacity.retry_if_result(lambda s: s is None),
                retry_error_callback=(lambda s: s.result())
                )
    def _broker_connect(self):
        """Tries to connect to MQTT broker until it succeeds"""
        logging.info("Attempting connexion to broker at {}:{}".format(self.config['broker_ip'], self.config['broker_port']))
        try:
            broker = mqtt.Client()
            broker.on_connect = self._on_broker_connect
            broker.connect(self.config['broker_ip'], int(self.config['broker_port']), 0)
            return broker
        except:
            logging.warning("Failed to connect to broker (Auto-retry)")
            return None

    def _on_broker_connect(self, client, userdata, flags, rc):
        logging.info("Succefully connected to broker")
        for topic in self.event_manifest["broker_message"].keys():
                self.broker.subscribe(topic)

    def _on_broker_message(self, client, userdata, message):
        msg = str(message.payload.decode("utf-8"))
        topic = message.topic
        try:
            payload = json.loads(msg)
        except:
            logging.warning("Failed to parse Json")
            payload = {}
        if 'value' in payload.keys():
            value = payload['value']
            if value not in self.event_manifest["broker_message"][topic].keys():
                value = 'any'
        else:
            value = 'any'
        logging.debug("Received message '{}' from topic {}".format(msg, topic))
        if topic in self.event_manifest["broker_message"]:
            if value in self.event_manifest["broker_message"][topic]:
                actions = self.event_manifest["broker_message"][topic][value]
                self.resolve_actions(actions)

    def resolve_actions(self, actions, value=None):
        for action in actions.keys():
            if action == "triggers":
                for trigger in actions['triggers']:
                    if trigger == "vad_start":
                        self.vadengine_thread.start_detecting()
                    if trigger == "vad_stop":
                        self.vadengine_thread.stop_detecting()
                    if trigger == "deactivate":
                        logging.debug("WUW is off")
                        self.provider_thread.provide_mfcc = False
                    if trigger == "activate":
                        logging.debug("WUW is on")
                        self.provider_thread.activate_mfcc_providing()
                    if trigger == "exit":
                        self.broker.disconnect()
            elif action == "publish":
                topic = actions["publish"]["topic"]
                msg = actions["publish"]["message"]
                msg = msg.replace("%(DATETIME)", datetime.datetime.now().isoformat())
                if value is not None:
                    msg = msg.replace("%(VALUE)", str(value))
                self.broker.publish(topic, msg)
                logging.debug("Published message '{}' on topic {}".format(msg, topic))

    def _on_event(self, event, value=None):
        logging.debug("Received event '{}' with value '{}'".format(event, value))
        if event in self.event_manifest['internal'].keys():
            actions = self.event_manifest['internal'][event]
            self.resolve_actions(actions, value)

def main():
    logging.basicConfig(level=logging.DEBUG, format="%(levelname)8s %(asctime)s [Commander] %(message)s ")

    # Read default config from file
    config = configparser.ConfigParser()
    config.read(os.path.dirname(os.path.abspath(__file__))+"/config.conf")
    runner = VoiceCommander(config)
    runner.run()

if __name__ == '__main__':
    main()
