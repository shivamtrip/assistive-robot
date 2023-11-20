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
        self.updateParamService = rospy.ServiceProxy('/update_param', UpdateParam)
        
        rospy.loginfo("Waiting for /startedListening service")
        self.startedListeningService.wait_for_service()
        
        rospy.loginfo("Waiting for /robot_task_command service")
        self.commandService.wait_for_service()

        rospy.loginfo("Waiting for /update_param service")
        self.updateParamService.wait_for_service()

        self.attention_sounds = ["Uh yes?", "Yes?", "What's up?", "How's life?", "Hey!", "Hmm?"]

        self.root = ""
        
        rospy.loginfo("HRI Node ready")

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
            phrase = random.choice(phrases)
            self.update_param(self.root + "hri_params/response", phrase)
            self.responseGenerator.run_tts(phrase)
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
        attn_sound = random.choice(self.attention_sounds)
        self.responseGenerator.run_tts(attn_sound)
        return attn_sound

    def update_param(self, path, value):
        req = UpdateParamRequest()
        req.path = path
        req.value = value
        self.updateParamService(req)

    def clear_params(self):
        self.update_param(self.root + "hri_params/wakeword", "")
        self.update_param(self.root + "hri_params/command", "")
        self.update_param(self.root + "hri_params/response", "")
        self.update_param(self.root + "hri_params/ack", "")

    def wakeword_triggered(self):
        self.clear_params()
        print("Wakeword triggered!")
        self.update_param(self.root + "hri_params/wakeword", "1")

        self.startedListeningService()
        self.speech_recognition.suppress_noise()
        ack = self.triggerWakewordThread()
        
        self.update_param(self.root + "hri_params/ack", ack)

        text = self.speech_recognition.speech_to_text()
        self.update_param(self.root + "hri_params/command", text)

        # send a trigger request to planning node saying that the wakeword has been triggered
        response, primitive = self.responseGenerator.processQuery(text)

        if primitive == "engagement" or primitive == "<none>":
            self.responseGenerator.run_tts(response)
            self.wakeword_detector.startRecorder()
            return
        elif primitive == "video_call_start":
            self.responseGenerator.run_tts("Starting a video call")
            self.update_param(self.root + "hri_params/response", "Starting a video call")
            self.update_param(self.root + "hri_params/operation_mode", "TELEOPERATION")
            return
        elif primitive == "video_call_stop":
            self.update_param(self.root + "hri_params/operation_mode", "AUTONOMOUS")
            return
        
        print(text, primitive, response)

        task = GlobalTaskRequest(speech = text, type = primitive, primitive = primitive)
        self.wakeword_detector.startRecorder()
        self.commandService(task)

        self.clear_params()


    def run(self):
        self.wakeword_detector.run()

if __name__ == "__main__":
    hri = HRI()
    hri.run()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
