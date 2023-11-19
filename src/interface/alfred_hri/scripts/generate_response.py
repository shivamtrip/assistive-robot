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
from alfred_msgs.msg import Speech, SpeechTrigger
from std_msgs.msg import String

# from response_generator.srv import VerbalResponse, VerbalResponseResponse

from difflib import ndiff
import json
import os
import requests
import subprocess
import sys
import time

from bs4 import BeautifulSoup
from google.cloud import texttospeech
import openai

class ResponseGenerator():
    def __init__(self):

        if not os.path.isfile(os.path.expanduser("~/.secrets.json")):
            sys.exit("ERROR: secrets.json does not exist. Please create it and add your access key.")
        secrets_file = os.path.expanduser("~/.secrets.json")
        config = {}

        with open(secrets_file, "r") as f:
            config = f.read()
            config = json.loads(config)

        self.openai_access_key = None
        if "OPENAI_API_KEY" not in config.keys():
            sys.exit("ERROR: OPENAI_API_KEY is not set. Please add it to ~/.secrets.json.")
        self.openai_access_key = config["OPENAI_API_KEY"]

        self.google_cloud_api_key = None
        if "GOOGLE_CLOUD_API_KEY" not in config.keys():
            sys.exit("ERROR: GOOGLE_CLOUD_API_KEY is not set. Please add it to ~/.secrets.json.")
        self.google_cloud_api_key = config["GOOGLE_CLOUD_API_KEY"]

        self.client = texttospeech.TextToSpeechClient()

        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
        }
        
        self.sounds = {}

        self.primitives_file = rospy.get_param("primitives_file_name", "resources/commands/primitives.json")
        self.primitives_file_path = os.path.join(rospkg.RosPack().get_path("alfred_hri"), self.primitives_file)

        self.primitives = json.load(open(self.primitives_file_path))
        
        uh_huh_file = rospy.get_param("uhuh_file", "resources/media/uh-huh.mp3")
        uh_huh_file = os.path.join(rospkg.RosPack().get_path("alfred_hri"), uh_huh_file)
        self.sounds["uh_huh"] = uh_huh_file
        self.commands = []
        self.match_command_to_primitive = {}
        for primitive in self.primitives.keys():
            for command in self.primitives[primitive]:
                self.commands.append(command)
                self.match_command_to_primitive[command] = primitive

        self.query_prefix = '''
        You are Alfred, a friendly indoor robot designed 
        to assist the elderly in nursing homes. Start acting like Alfred. You don't need to introduce yourself. Respond concisely.
        '''
        self.history = []

    def playSound(self, phrase = 'uh_huh'):
        sound_file = self.sounds[phrase]
        rospy.loginfo(f"Playing {sound_file}")
        self.play_sound_file(sound_file)

    def play_sound_file(self, file):
        # Play a sound file
        os.system(f"aplay --nonblock {file}")
        # pass


    def weather(self, city):
        city = city.replace(" ", "+")
        res = requests.get(
            f'https://www.google.com/search?q={city}&oq={city}&aqs=chrome.0.35i39l2j0l4j46j69i60.6128j1j7&sourceid=chrome&ie=UTF-8', headers=self.headers)
        print("Searching...\n")
        soup = BeautifulSoup(res.text, 'html.parser')
        location = soup.select('#wob_loc')[0].getText().strip()
        time = soup.select('#wob_dts')[0].getText().strip()
        info = soup.select('#wob_dc')[0].getText().strip()
        weather = soup.select('#wob_tm')[0].getText().strip()
        
        return weather

    def run_tts(self, text):
        # Run Google Cloud TTS
        synthesis_input = texttospeech.SynthesisInput(text=text)

        voice = texttospeech.VoiceSelectionParams(
            language_code="en-US", ssml_gender=texttospeech.SsmlVoiceGender.MALE, name="en-US-Neural2-I"
        )

        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16
        )

        response = self.client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )

        sound_file = "/tmp/tts_output.mp3"

        with open(sound_file, "wb+") as out:
            # Write the response to the output file.
            out.write(response.audio_content)
            print(f'Audio content written to file "{sound_file}"')

        # Play the audio file
        self.play_sound_file(sound_file)


    def process_gpt_query(self, query):
        openai.api_key = self.openai_access_key

        query = self.query_prefix + query
        user_dict = {"role": "user", "content": query}
        self.history.append(user_dict)

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": self.query_prefix},
                *self.history
                
            ],
        )
        assistant_dict = {"role": "assistant", "content": response.choices[0].message.content}
        self.history.append(assistant_dict)
        return response.choices[0].message.content
    
    def processQuery(self, text):
        if text is not None:
            rospy.loginfo("Heard: " + text)
            
            closest_command = difflib.get_close_matches(text, self.commands, n=1, cutoff=0.75)
            if len(closest_command) > 0:
                rospy.loginfo("Closest command: " + closest_command[0])
                cmd = closest_command[0]
                primitive = self.match_command_to_primitive[cmd]
                response = 'On it!'
            else:
                rospy.loginfo("Not in diction. Offloading to GPT.")
                response = self.process_gpt_query(text)
                primitive = 'engagement'
        else:
            response = "Sorry, I didn't catch that."
            primitive = '<none>'

        return (response, primitive)
    
    def similar(self, str1, str2):
        len1 = len(str1) + 1
        len2 = len(str2) + 1

        col = [0,] * len1
        prevCol = [0,] * len1

        for i in range(len1):
            prevCol[i] = i
        
        for i in range(len2):
            col[0] = i
            for j in range(1, len1):
                col[j] = min(min(1+col[j-1], 1+prevCol[j]),
                        prevCol[j-1] + (str1[j-1] != str2[i-1]))
            col, prevCol = prevCol, col
        
        dist = prevCol[len1-1]

        return 1 - (dist / max(len(str1), len(str2)))

    
    def engagement_callback(self, data : str):
        if ("weather" in data.lower()):
            response = "The weather outside is " + str(self.weather("Pittsburgh")) + \
                    " degrees fahrenheit in Pittsburgh."
        elif (self.similar(data.lower(), "thank you") > 0.8 or self.similar(data.lower(), "thanks") > 0.8):
            response = 'You\'re welcome!'

        elif (self.similar(data.lower(), "who created you") > 0.8 or self.similar(data.lower(), "who made you") > 0.8):
            response = "I was created by a group of graduate students at the Carnegie Mellon Robotics Institute."

        elif (self.similar(data.lower(), "who are you") > 0.8):
            response = "I am Alfred, a robot developed to help the elderly in nursing homes."

        elif (self.similar(data.lower(), "what can you do")) > 0.8:
            response = "I can help you fetch water, tell you a joke, and tell you the weather."

        elif (self.similar(data.lower(), "how affordable are you") > 0.8):
            response = "I am very affordable. Just don't add any more Apple products to my body and keep Atharva away from the order list. You should be fine."

        elif (self.similar(data.lower(), "how good is our business model") > 0.8):
            response = "You guys talk so much about A I. You should start predicting your bankruptcy date. Might as well slap an M L model on it."
        else:
            response = self.process_gpt_query(data.lower())

        print("Response:", response)
        self.run_tts(response)

if __name__ == "__main__":
    rospy.init_node("response_generator_node")
    response_generator = ResponseGenerator()
    response_generator.run()
