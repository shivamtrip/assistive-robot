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

from task_planner.msg import PlanTriggerAction, PlanTriggerGoal, PlanTriggerResult, PlanTriggerFeedback


from firebase_node import FirebaseNode
import actionlib
class HRI():
    def __init__(self):
        rospy.init_node("alfred_hri")
        
        self.firebase_node = FirebaseNode()
        
        self.speech_recognition = SpeechRecognition()
        self.wakeword_detector = WakewordDetector(self.wakeword_triggered)
        self.responseGenerator = ResponseGenerator()
        self.verbal_response_service = rospy.Service(
            '/interface/response_generator/verbal_response_service', 
            VerbalResponse, 
            self.verbal_response_callback
        )
        self.update_param = self.firebase_node.update_param
        
        self.startedListeningService = rospy.ServiceProxy('/startedListening', Trigger)
        # self.commandService = rospy.ServiceProxy('/robot_task_command', GlobalTask)
        self.command_action_client = actionlib.SimpleActionClient('/robot_task_command', PlanTriggerAction)
        self.updateParamService = rospy.Service('/update_param', UpdateParam, self.firebase_node.update_param)
        
        rospy.loginfo("Waiting for /startedListening service")
        self.startedListeningService.wait_for_service()
        
        rospy.loginfo("Waiting for /robot_task_command server")
        self.command_action_client.wait_for_server()

        self.attention_sounds = ["Uh yes?", "Yes?", "What's up?", "How's life?", "Hey!", "Hmm?"]
        rospy.loginfo("HRI Node ready")

        self.code = ""

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
            self.update_param("hri_params/response", phrase)
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

    def clear_params(self):
        self.update_param("hri_params/wakeword", "")
        self.update_param("hri_params/command", "")
        self.update_param("hri_params/response", "")
        self.update_param("hri_params/ack", "")

    def wakeword_triggered(self):
        self.clear_params()
        self.update_param("hri_params/wakeword", "1")

        self.startedListeningService()
        self.speech_recognition.suppress_noise()
        ack = self.triggerWakewordThread()
        
        self.update_param("hri_params/ack", ack)

        text = self.speech_recognition.speech_to_text()
        self.update_param("hri_params/command", text)

        # send a trigger request to planning node saying that the wakeword has been triggered
        response, primitive = self.responseGenerator.processQuery(text)

        # if primitive == "engagement" or primitive == "<none>":
        #     self.responseGenerator.run_tts(response)
        #     self.wakeword_detector.startRecorder()
        #     return
        # elif primitive == "video_call_start":
        #     self.responseGenerator.run_tts("Starting a video call")
        #     self.update_param("hri_params/response", "Starting a video call")
        #     self.update_param("hri_params/operation_mode", "TELEOPERATION")
        #     return
        # elif primitive == "video_call_stop":
        #     self.update_param("hri_params/operation_mode", "AUTONOMOUS")
        #     return
        if primitive == "code":
            self.code = response
            self.wakeword_detector.startRecorder()
            return
        elif primitive == "summary":
            self.responseGenerator.run_tts(response)
            self.wakeword_detector.startRecorder()
            return
        elif primitive == "affirm":
            phrases = ["Ok", "Okay", "Sure", "Alright", "I'll do that"]
            self.responseGenerator.run_tts(random.choice(phrases))
        elif primitive == "negate":
            self.code = ""
            phrases = ["Closing the task", "Ok, I won't do that", "Ok, I'll stop", "Ok, I'll stop doing that"]
            self.responseGenerator.run_tts(random.choice(phrases))
            self.wakeword_detector.startRecorder()
            return
        else:
            self.wakeword_detector.startRecorder()
            return
        
        print(text, primitive, response)
        
        task = PlanTriggerGoal(plan = self.code )
        self.command_action_client.send_goal(task)
        self.wakeword_detector.startRecorder()
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
