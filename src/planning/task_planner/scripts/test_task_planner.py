from task_planner import TaskPlanner
from state_manager import *
import rospy
task_planner = TaskPlanner()
# task_planner.navigate_to_location(LocationOfInterest.LIVING_ROOM)
task_planner.task_requested = True
task_planner.navigationGoal = LocationOfInterest.KITCHEN
task_planner.objectOfInterest = ObjectOfInterest.BOTTLE
task_planner.executeTask()
try:
    rospy.spin()
except KeyboardInterrupt:
    print("Shutting down")
