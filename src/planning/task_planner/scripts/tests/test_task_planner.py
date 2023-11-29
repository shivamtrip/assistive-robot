from task_planner.msg import PlanTriggerAction, PlanTriggerGoal, PlanTriggerResult, PlanTriggerFeedback
import rospy
import actionlib
"""
Hey, Alfred! Go to the drawer. 
Whatever you find above the drawer, put it inside. 
Once you do that, go to the table near the sofa. 
You might find a teddy bear there. 
If you find it, bring it back to me (user) via the route 
between potted plant and sofa, and place it near me (user).
"""
goal = PlanTriggerGoal()
goal.plan = """
self.go_to("drawer")
self.open_drawer()
detections = self.get_detections()
for detection in detections:
    self.pick(detection)
    self.place("drawer")
self.close_drawer()

self.move_object_closest_to("table", "sofa")
success = self.find_and_align_to_object("teddy_bear")
if success:
    self.pick("teddy_bear")
    self.move_between_objects("potted plant", "sofa")
    self.go_to("user")
    self.place("table")
    self.speak("I'm done.")
else:
    self.speak("I can't find the teddy bear.")
"""
rospy.init_node("temp")
rospy.loginfo("Waiting for plan_trigger server...")
client = actionlib.SimpleActionClient('task_planner', PlanTriggerAction)
client.wait_for_server()

client.send_goal(goal)
client.wait_for_result()
