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
        self.m = sr.Microphone()
        rospy.loginfo("Speech recognition adjusting for ambient noise")
        with self.m as source:
            self.r.adjust_for_ambient_noise(source, duration=5)

        rospy.loginfo("Set minimum energy threshold to {}".format(self.r.energy_threshold))
        

    def speech_to_text(self):
        # Record audio
        with self.m as source:
            audio = self.r.listen(source, timeout=5, phrase_time_limit=5)

        # Convert audio to text
        try:
            text = self.r.recognize_google(audio)
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
