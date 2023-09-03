
import actionlib
import rospy
from manipulation.msg import TriggerAction, TriggerFeedback, TriggerResult, TriggerGoal
from std_srvs.srv import Trigger, TriggerResponse
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal, MoveBaseActionResult, MoveBaseResult
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../scripts/'))

class TaskPlannerTest:

    def __init__(self):
        # self.startManipService = rospy.ServiceProxy('/switch_to_manipulation_mode', Trigger)
        # self.startNavService = rospy.ServiceProxy('/switch_to_navigation_mode', Trigger)

        # self.navigation_client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        # self.manipulation_client = actionlib.SimpleActionClient('manipulation_fsm', TriggerAction)

        # self.verbal_response_service_name = rospy.get_param("verbal_response_service_name",
        #                         "/interface/response_generator/verbal_response_service")

        # self.startedListeningService = rospy.Service('/startedListening', Trigger, self.wakeWordTriggered)
        # self.commandReceived = rospy.Service('/robot_task_command', GlobalTask, self.command_callback)
        pass


if __name__ == "__main__":
    from task_planner import TaskPlanner