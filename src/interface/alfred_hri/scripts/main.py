#!/usr/local/lib/robot_env/bin/python3

import json
import os
import random
from threading import Thread

import rospy
from std_srvs.srv import Trigger
from alfred_msgs.srv import VerbalResponse, VerbalResponseRequest, VerbalResponseResponse, GlobalTask, GlobalTaskResponse, GlobalTaskRequest, UpdateParam, UpdateParamRequest, UpdateParamResponse

import pyrebase

from generate_response import ResponseGenerator
from recognize_speech import SpeechRecognition
from wakeword_detector import WakewordDetector

class HRI():
    def __init__(self):
        rospy.init_node("alfred_hri")
        self.speech_recognition = SpeechRecognition()
        self.wakeword_detector = WakewordDetector(self.wakeword_triggered)
        self.responseGenerator = ResponseGenerator()
        self.verbal_response_service = rospy.Service(
            '/interface/response_generator/verbal_response_service', 
            VerbalResponse, 
            self.verbal_response_callback
        )

        self.startedListeningService = rospy.ServiceProxy('/startedListening', Trigger)
        self.commandService = rospy.ServiceProxy('/robot_task_command', GlobalTask)
        
        rospy.loginfo("Waiting for services")
        self.startedListeningService.wait_for_service()
        self.commandService.wait_for_service()

        rospy.loginfo("Waiting for update_param service")
        self.updateParamService = rospy.ServiceProxy('/update_param', UpdateParam)
        self.updateParamService.wait_for_service()

        self.attention_sounds = ["uh yes?", "yes?", "what's up?", "how's life?", "hey!", "hmm?"]

        rospy.loginfo("HRI Node ready")

        firebase_secrets_path = os.path.expanduser("~/.alfred-auxilio-firebase-adminsdk.json")

        if not os.path.isfile(firebase_secrets_path):
            raise FileNotFoundError("Firebase secrets file not found")
        
        with open(firebase_secrets_path, 'r') as f:
            config = json.load(f)

        self.firebase = pyrebase.initialize_app(config)
        self.db = self.firebase.database()

    def verbal_response_callback(self, req : VerbalResponseRequest):
        # call the response generator to generate a response
        if (req.response == "on_it"):
            phrases = [
                "On it",
                "I'll get right on it", 
                "I'll do it right away", 
                "I'll do it right now",
                "Let me get that for you"
            ]
            self.responseGenerator.run_tts(random.choice(phrases))
        elif (req.response == "ok"):
            phrases = ["Ok", "Okay", "Sure", "Alright", "I'll do that"]
            self.responseGenerator.run_tts(random.choice(phrases))
        elif (req.response == "here_you_go"):
            phrases = ["Here you go", "Here you are", "Here", "Here it is", "Here's your thing"]
            self.responseGenerator.run_tts(random.choice(phrases))
        else:
            self.responseGenerator.run_tts(req.response)
        return VerbalResponseResponse(status = "success")

    def triggerWakewordThread(self):
        self.responseGenerator.run_tts(random.choice(self.attention_sounds))

    def update_param(self, path, value):
        req = UpdateParamRequest()
        req.path = path
        req.value = value
        self.updateParamService(req)

    def wakeword_triggered(self):
        print("Wakeword triggered!")
        root = ""
        self.update_param("hri_params/wakeword", "1")

        self.startedListeningService()
        self.triggerWakewordThread()
        self.update_param("hri_params/ack", "1")

        text = self.speech_recognition.speech_to_text()
        self.update_param("hri_params/command", text)

        # send a trigger request to planning node saying that the wakeword has been triggered
        response, primitive = self.responseGenerator.processQuery(text)
        self.update_param("hri_params/response", response)

        if primitive == "engagement" or primitive == "<none>":
            self.responseGenerator.run_tts(response)
            self.wakeword_detector.startRecorder()
            return
        print(text, primitive, response)

        task = GlobalTaskRequest(speech = text, type = primitive, primitive = primitive)
        self.wakeword_detector.startRecorder()
        self.commandService(task)

        self.update_param("hri_params/wakeword", "")
        self.update_param("hri_params/command", "")
        self.update_param("hri_params/response", "")
        self.update_param("hri_params/ack", "")


    def run(self):
        self.wakeword_detector.run()



if __name__ == "__main__":
    hri = HRI()
    hri.run()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
