#!/usr/local/lib/robot_env/bin/python3

#
# Copyright (C) 2023 Auxilio Robotics
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#

# ROS imports
import rospkg
import rospy
import rostopic

from alfred_msgs.msg import SpeechTrigger
from std_msgs.msg import String

import json
import os
import subprocess
import sys

import pvporcupine
from pvrecorder import PvRecorder
from std_srvs.srv import Trigger

class WakewordDetector():
    def __init__(self, triggerCallback):
        self.wakeword = rospy.get_param("wakeword", "Hey Alfred")
        self.keywords_path = rospy.get_param("keywords_path", "resources/keywords/Hey-Alfred_en_linux_v2_1_0.ppn")
        self.models_path = rospy.get_param("models_path", "resources/models/porcupine_params.pv")
        self.audio_device_index = rospy.get_param("audio_device_index", 1)
        self.rospack = rospkg.RosPack()
        self.package_path = self.rospack.get_path("alfred_hri")

        self.keyword_paths = [os.path.join(self.package_path, self.keywords_path)]
        self.models_path = os.path.join(self.package_path, self.models_path)
        self.keyword_sensitivities = [0.5]

        # Access key is stored in ~/.secrets.sh
        self.triggerCallback = triggerCallback
        # check if file exists
        if not os.path.isfile(os.path.expanduser("~/.secrets.json")):
            print(os.path.expanduser('~/.secrets.json'))
            sys.exit("ERROR: secrets.json does not exist. Please create it and add your access key.")
        
        secrets_file = os.path.expanduser("~/.secrets.json")
        config = {}
        with open(secrets_file, "r") as f:
            config = f.read()
            config = json.loads(config)

        self.access_key = None

        if "PORCUPINE_ACCESS_KEY" not in config.keys():
            sys.exit("ERROR: PORCUPINE_ACCESS_KEY is not set. Please add it to ~/.secrets.json.")

        self.access_key = config["PORCUPINE_ACCESS_KEY"]

        self.porcupine = pvporcupine.create(
                access_key=self.access_key,
                keyword_paths=self.keyword_paths,
                sensitivities=self.keyword_sensitivities,
                model_path=self.models_path)

        self.recorder = PvRecorder(
                device_index=self.audio_device_index,
                frame_length=self.porcupine.frame_length,
        )
        self.recorderStarted = True
        self.recorder.start()

    def process_audio(self, in_data):
        keyword_index = self.porcupine.process(in_data)
        if keyword_index >= 0:
            print("Detected keyword: " + self.wakeword) 
            self.recorder.stop()
            self.recorderStarted = False
            self.triggerCallback()

    def startRecorder(self):
        self.recorderStarted = True
        self.recorder.start()
            

    def run(self):
        while True:
            if self.recorderStarted:
                self.process_audio(self.recorder.read())


if __name__ == "__main__":
    def dumbcallbac():
        pass
    wakeword_detector = WakewordDetector(dumbcallbac)
    wakeword_detector.run()
