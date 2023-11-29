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
import rospy
import rospkg
import rostopic

import difflib
import json
import os
import sys
import time

import speech_recognition as sr
from std_srvs.srv import Trigger, TriggerResponse
class SpeechRecognition():
    def __init__(self):
        self.r = sr.Recognizer()
        # rospy.loginfo("Speech recognition adjusting for ambient noise")

        secrets_file = os.path.expanduser("~/.secrets.json")
        config = {}
        with open(secrets_file, "r") as f:
            config = f.read()
            config = json.loads(config)

        self.openai_access_key = None

        if "OPENAI_API_KEY" not in config.keys():
            sys.exit("ERROR: OPENAI_API_KEY is not set. Please add it to ~/.secrets.json.")

        self.openai_access_key = config["OPENAI_API_KEY"]

        # rospy.loginfo("Set minimum energy threshold to {}".format(self.r.energy_threshold))

    def suppress_noise(self):
        m = sr.Microphone()

        with m as source:
            self.r.adjust_for_ambient_noise(source, duration=0.5)
            rospy.loginfo("Set minimum energy threshold to {}".format(self.r.energy_threshold))

    def speech_to_text(self):
        # Record audio

        m = sr.Microphone()
        with m as source:
            print("Listening...")
            audio = self.r.listen(source, timeout=5, phrase_time_limit=20)
        
        rospy.loginfo("Recognized speech. Transcribing...")

        # Convert audio to text
        try:
            text = self.r.recognize_whisper_api(audio, api_key=self.openai_access_key)
            return text
        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand audio")
        except sr.RequestError as e:
            print("Could not request results from Google Speech Recognition service; {0}".format(e))

        return None

    

if __name__ == "__main__":
    rospy.init_node("speech_recognition_node")
    speech_recognition = SpeechRecognition()
    rospy.spin()
