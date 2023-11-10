

from enum import Enum


class TaskType(Enum):

    DELIVERY = 0
    VIDEOCALL = 1


class Emotions(Enum):
    HAPPY = 0
    SAD = 1
    ANGRY = 2
    DROWSY = 3
    BORED = 4
    NEUTRAL = 5
    ATTENTION = 6

class OperationModes(Enum):
    TELEOPERATION = 0
    AUTONOMOUS = 1

class GlobalStates(Enum):
    IDLE = 10
    BOOTING = 0
    WAITING_FOR_COMMAND = 1
    RECEIVED_COMMAND = 2
    NAVIGATING = 3
    REACHED_GOAL = 4
    RECOVERY = 9
    MANIPULATION = 6
    COMPLETED_TASK = 7
    CALL = 8
    

class ObjectOfInterest(Enum): 
    # FILL WITH CLASS LABELS FOR THE OBJECT DETECTION PIPELINE
    NONE = -1
    USER = 0
    BOTTLE = 39
    BOX = 2
    GLASS = 3
    TABLE = 60
    REMOTE = 65
    APPLE = 47
    BANANA = 46
    

class LocationOfInterest(Enum):
    # FILL WITH LOCATION LABELS FROM GROUNDING PIPELINE
    HOME = -1
    LIVING_ROOM = 0
    KITCHEN = 1
    TABLE = 2
    NET = 3

class VerbalResponseStates(Enum):
    # FILL WITH VERBAL RESPONSES
    NONE = -1
    UHUH = 0
    OK = 1
    ON_IT = 2
    SORRY = 3
    THANKS = 4
    YES = 5
    HERE_YOU_GO = 6


