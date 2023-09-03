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

from alfred_msgs.msg import Speech, SpeechTrigger
from std_msgs.msg import String

import difflib
import json
import os
import sys
import time

import speech_recognition as sr
from std_srvs.srv import Trigger, TriggerResponse
class SpeechRecognition():
    def __init__(self):
        self.wakeword_topic_name = rospy.get_param("wakeword_topic_name",
                "/interface/wakeword_detector/trigger")
        # self.wakeword_subscriber = rospy.Subscriber(self.wakeword_topic_name, SpeechTrigger, self.wakeword_callback)
        self.wakeword_subscriber = rospy.Service('wakeword_trigger', Trigger, self.wakeword_callback)

        self.command_topic_name = rospy.get_param("command_topic_name",
                "/interface/speech_recognition/command")
        self.command_publisher = rospy.Publisher(self.command_topic_name, Speech, queue_size=10)

        self.engagement_topic_name = rospy.get_param("engagement_topic_name",
                "/interface/speech_recognition/engagement")
        self.engagement_publisher = rospy.Publisher(self.engagement_topic_name, Speech, queue_size=10)

        self.r = sr.Recognizer()

        self.prmitives_file = rospy.get_param("primitives_file_name", "primitives.json")
        self.prmitives_file_path = os.path.join(rospkg.RosPack().get_path("speech_recognition"), "commands", self.prmitives_file)

        self.primitives = json.load(open(self.prmitives_file_path))

        self.match_command_to_primitive = {}
        self.commands = []

        for primitive in self.primitives.keys():
            for command in self.primitives[primitive]:
                self.commands.append(command)
                self.match_command_to_primitive[command] = primitive

    def speech_to_text(self):
        # Record audio
        with sr.Microphone() as source:
            audio = self.r.listen(source)

        # Convert audio to text
        try:
            text = self.r.recognize_google(audio)
            return text
        except speech_recognition.UnknownValueError:
            print("Google Speech Recognition could not understand audio")
        except speech_recognition.RequestError as e:
            print("Could not request results from Google Speech Recognition service; {0}".format(e))
        return None

    def wakeword_callback(self, msg):
        # Record audio
        rospy.loginfo("Listening for commands...")
        text = self.speech_to_text()

        if text != None:
            rospy.loginfo("Heard: " + text)
            closest_command = difflib.get_close_matches(text, self.commands, n=1, cutoff=0.75)
            if len(closest_command) > 0:
                rospy.loginfo("Closest command: " + closest_command[0])
                cmd = closest_command[0]
                primitive = self.match_command_to_primitive[cmd]
                msg = Speech()
                msg.type = "command"
                msg.speech = text
                msg.primitive = primitive
                self.command_publisher.publish(msg)
            else:
                rospy.loginfo("Setting engagement mode...")
                msg = Speech()
                msg.type = "engagement"
                msg.speech = text
                msg.primitive = "null"
                self.engagement_publisher.publish(msg)
        return TriggerResponse(success = True)

if __name__ == "__main__":
    rospy.init_node("speech_recognition_node")
    speech_recognition = SpeechRecognition()
    rospy.spin()
