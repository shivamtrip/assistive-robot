

from enum import Enum



class BotState:
    def __init__(self):
        self.current_global_state = GlobalStates.IDLE
        self.operation_mode = OperationModes.AUTONOMOUS
        self.emotion = Emotions.NEUTRAL
        self.battery_percent = None
        self.music = None



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
    NAVIGATION = 3
    MANIPULATION = 6
    VIDEOCALL = 5
    

class ObjectOfInterest(Enum): 
    BOTTLE = 39
    

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


