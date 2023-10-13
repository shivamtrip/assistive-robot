import stretch_body.robot as rb
import time
import numpy as np

robot = rb.Robot()
if not robot.startup():
    exit() # failed to start robot!
# robot.home()
# cpos = 30.0
# cneg = -30.0
# print("Sending command")
# robot.lift.move_to(0.6)




robot.arm.move_to(0.2)
robot.push_command()
robot.head.move_to("head_pan", -np.pi/2)
robot.head.move_to("head_tilt", -45 * np.pi/180)
robot.end_of_arm.move_to("wrist_yaw", 90 * np.pi/180)  
robot.push_command()
cpos = 30
cneg = -30
robot.lift.move_to(0.75)
robot.push_command()
robot.arm.wait_until_at_setpoint(timeout=5.0)
robot.lift.wait_until_at_setpoint(timeout=5.0)
robot.arm.move_to(0.5, contact_thresh_pos=cpos, contact_thresh_neg=cneg)
robot.push_command()
robot.arm.wait_until_at_setpoint(timeout=5.0)

cpos = 30.0
cneg = -30.0
robot.lift.move_by(-0.08, contact_thresh_pos=cpos, contact_thresh_neg=cneg)
robot.push_command()
robot.lift.wait_until_at_setpoint(timeout=5.0)

cpos = 30
cneg = -30
robot.arm.move_to(0.1, contact_thresh_pos=cpos, contact_thresh_neg=cneg)
robot.push_command()
robot.arm.wait_until_at_setpoint(timeout=5.0)


robot.stop()


