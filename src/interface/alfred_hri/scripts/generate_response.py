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

        # self.primitives_file = rospy.get_param("primitives_file_name", "resources/commands/primitives.json")
        # self.primitives_file_path = os.path.join(rospkg.RosPack().get_path("alfred_hri"), self.primitives_file)

        # self.primitives = json.load(open(self.primitives_file_path))
        
        uh_huh_file = rospy.get_param("uhuh_file", "resources/media/uh-huh.mp3")
        uh_huh_file = os.path.join(rospkg.RosPack().get_path("alfred_hri"), uh_huh_file)
        self.sounds["uh_huh"] = uh_huh_file
        # self.commands = []
        # self.match_command_to_primitive = {}
        # for primitive in self.primitives.keys():
        #     for command in self.primitives[primitive]:
        #         self.commands.append(command)
        #         self.match_command_to_primitive[command] = primitive

        self.query_prefix = """
            Available robot functions:\n
            self.go_to('object'), self.move_between_objects('object1', 'object2'), self.move_object_closest_to('object1', 'object2'), self.pick('object'), self.place('surface_name'), self.open_drawer(), self.close_drawer(), self.find_and_align_to_object('object'), self.get_detections(), speak(text) \n
            Objects in the scene: potted_plant, sofa, apple, banana, remote, water_bottle, teddy_bear\n
            Docstrings are as follows -\n
            def go_to(object):
                Moves the robot to the object location
                Parameters:
                object (string): Object name

            def move_between_objects(object1, object2):
                Moves the robot to a location between object1 and object2
                Parameters:
                object1 (string): Object 1 name
                object2 (string): Object 2 name

            def move_object_closest_to(object1, object2):
                Moves the robot close to object1 type that is closest to object2 type
                Parameters:
                object1 (string): Object 1 name
                object2 (string): Object 2 name

            def pick(object):
                Makes the robot pick up an object
                Parameters:
                object (string): Object name
            
            def place(surface):
                Makes the robot place the object that it is holding on to the target surface
                Parameters:
                surface (string): Surface name on which the object is to be placed

            def open_drawer():
                Makes the robot open a nearby drawer

            def close_drawer():
                Makes the robot close a nearby drawer

            def find_and_align_to_object(object):
                Makes the robot find an object nearby and align itselt to the object
                Parameters:
                object (string): Object name

            def get_detections():
                Returns an array of nearby objects that are currently being detected by the robot
                Returns:
                List: Array of detected object names as strings

            def speak(text):
                Makes the robot speak the given text input using a speaker
                Parameters:
                text (string): Text to be spoken by the robot
            
            Always generate commands on a new line\n
            Single quotes within a string are preceded by a back slash - \'. Also, don't use double quotes and take care of python indentation (4 spaces) \n
            If the user just says yes or affirms your code, just reply with a Yes, if they negate, just reply with a No\n
            If the user asks for a code sumamry, start reply with Summary: \n
            Note that there is always a table close to the user\n
            Command: Yes
            Answer:\n
            Yes
            Command: That sounds about right
            Answer:\n
            Yes
            Command: That is correct
            Answer:\n
            Yes
            Command: No
            Answer:\n
            No
            Command: That doesn't sound right
            Answer:\n
            No
            Command: That is not correct
            Answer:\n
            No
            Command: Bring me an apple\n
            Answer:\n
            self.go_to('apple')
            success = self.find_and_align_to_object('apple')
            if success:
                self.pick('apple')
                self.go_to('user')
                self.place('table')
                self.speak('Enjoy your apple')
            else:
                self.go_to('user')
                self.speak('I couldn't find you an apple')
            Command: Bring me a banana\n
            Answer:\n
            self.go_to('banana')
            success = self.find_and_align_to_object('banana')
            if success:
                self.pick('banana')
                self.go_to('user')
                self.place('table')
                self.speak('Enjoy your banana! Don\'t let it sit!')
            else:
                self.go_to('user')
                self.speak('I couldn't find you an banana')
            Command: Move between sofa and potted plant and then move to the table closest to the sofa
            Answer:\n
            self.move_between_objects('sofa', 'potted_plant')
            self.move_object_closest_to('table', 'sofa')
            Command: Crack a joke
            Answer:\n
            self.speak('Why didn't the skeleton go to the party? Because he had nobody to go with!')
            Command: Go to the drawer and place all items that you may find above it and place them all inside the drawer. Come back to me (user) and then crack a joke
            Answer:\n
            self.go_to('drawer')
            self.open_drawer()
            detections = self.get_detections()
            for detection in detections:
                self.pick(detection)
                self.place('drawer')
            self.close_drawer()
            self.go_to('user')
            self.speak('Why don't scientists trust atoms? Because they make up everything!')
            Command: Go to the potted plant. You may find a teddy bear near it. If you find it, bring it back to me.
            Answer:\n
            self.go_to('potted_plant')
            success = self.find_and_align_to_object('teddy_bear')
            if success:
                self.pick('teddy_bear')
                self.go_to('user')
                self.place('table')
                self.speak('Here's your teddy bear. Have fun cuddling!')
             else:
                self.go_to('user')
                self.speak('Sorry, I couldn't find your teddy bear')
            """
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
        user_dict = {"role": "user", "content": f"Command: {query}. Answer:\n"}
        self.history.append(user_dict)

        rospy.loginfo("Requesting GPT-4 completion")
        response = openai.ChatCompletion.create(
            model="gpt-4",
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
            # rospy.loginfo("Heard: " + text)
            
            # closest_command = difflib.get_close_matches(text, self.commands, n=1, cutoff=0.85)
            # if len(closest_command) > 0:
            #     rospy.loginfo("Closest command: " + closest_command[0])
            #     cmd = closest_command[0]
            #     primitive = self.match_command_to_primitive[cmd]
            #     response = 'On it!'
            # else:
            #     rospy.loginfo("Not in diction. Offloading to GPT.")
            #     response = self.process_gpt_query(text)
            #     primitive = 'engagement'
            response = self.process_gpt_query(text)
            if response == "Yes":
                primitive = "affirm"
            elif response == "No":
                primitive = "negate"
            elif "summary" in response.lower():
                primitive = "summary"
            else:
                primitive = "code"
        else:
            response = "Sorry, I didn't catch that."
            primitive = '<none>'
        rospy.loginfo("Response: " + response)
        rospy.loginfo("Primitive: " + primitive)
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
