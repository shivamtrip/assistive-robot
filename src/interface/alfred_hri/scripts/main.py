#!/usr/local/lib/robot_env/bin/python3

import random
from threading import Thread
from recognize_speech import SpeechRecognition
from wakeword_detector import WakewordDetector
from generate_response import ResponseGenerator
import rospy
from std_srvs.srv import Trigger
from alfred_msgs.srv import VerbalResponse, VerbalResponseRequest, VerbalResponseResponse, GlobalTask, GlobalTaskResponse, GlobalTaskRequest
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
        rospy.loginfo("HRI Node ready")
        self.attention_sounds = ["uh yes?", "yes?", "what's up?", "how's life?", "hey!", "hmm?"]

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

    def wakeword_triggered(self):
        print("Wakeword triggered!")
        self.startedListeningService()
        # thread = Thread(target=self.triggerWakewordThread)
        # thread.start()
        self.triggerWakewordThread()
        text = self.speech_recognition.speech_to_text()
        rospy.sleep(1.5)
        self.wakeword_detector.startRecorder()
        # send a trigger request to planning node saying that the wakeword has been triggered
        response, primitive = self.responseGenerator.processQuery(text)
        if primitive == "engagement" or primitive == "<none>":
            self.responseGenerator.run_tts(response)
            self.wakeword_detector.startRecorder()
            return
        print(text, primitive, response)
        task = GlobalTaskRequest(speech = text, type = primitive, primitive = primitive)
        self.wakeword_detector.startRecorder()
        self.commandService(task)
    def run(self):
        self.wakeword_detector.run()



if __name__ == "__main__":
    hri = HRI()
    hri.run()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")