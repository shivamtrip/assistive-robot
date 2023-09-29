#!/usr/bin/env python3

import time
import stretch_body.robot
import numpy as np


robot = stretch_body.robot.Robot()
robot.startup()
# robot.base.left_wheel.disable_sync_mode()
robot.lift.move_to(0.9)
robot.arm.move_to(1)

# robot.base.right_wheel.disable_sync_mode()
# robot.end_of_arm.move_to("stretch_gripper", -50) 
# robot.arm.move_to(0.01)
# robot.push_command()
# time.sleep(4)
robot.end_of_arm.move_to('stretch_gripper', 50)
robot.push_command()
time.sleep(3)
robot.end_of_arm.move_to('stretch_gripper', -30)
robot.push_command()
# robot.head.move_to('head_tilt', -np.pi/2.15)
# robot.head.move_to('head_pan', -np.pi/2.15)
# robot.base.rotate_by(-1.57)
# robot.push_command()
# time.sleep(4)
# robot.head.move_to('head_pan', 0)
# robot.head.move_to('head_tilt', 0)
# # robot.pretty_print()
# robot.home()
# robot.arm.set_velocity
# robot.base.translate_by(0.1)

# robot.base.rotate_by(1.57)

time.sleep(4)
print("DONE")
# print()
robot.stop()
