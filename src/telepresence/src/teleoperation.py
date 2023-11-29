import rospy


class TeleopController:
    
    
    def __init__(self, trajectoryClient, visual_servoing):
        
        self.trajectoryClient = trajectoryClient
        self.visual_servoing = visual_servoing
        self.teleop_commands = None
        


    def maintainvelocities(self):
        
        if self.teleop_commands:
            
            if self.teleop_commands['mobile_base']['velx'] == 0:
                self.visual_servoing.move_to_pose(self.trajectoryClient, 
                    {'base_rotate;vel' : self.teleop_commands['mobile_base']['veltheta']},
                )
            else:
                self.visual_servoing.move_to_pose(self.trajectoryClient, 
                    {'base_translate;vel' : self.teleop_commands['mobile_base']['velx']},
                )
            gripperpos = -50
            if self.teleop_commands['manipulator']['gripper_open']:
                gripperpos = 100
            self.visual_servoing.move_to_pose(self.trajectoryClient, 
                {'stretch_gripper;to' : gripperpos},
            )
            self.visual_servoing.move_to_pose(self.trajectoryClient, 
                {'wrist_yaw;to' : self.teleop_commands['manipulator']['yaw_position']},
            )
            self.visual_servoing.move_to_pose(self.trajectoryClient, 
                {'lift;vel' : self.teleop_commands['manipulator']['vel_lift']},
            )
            self.visual_servoing.move_to_pose(self.trajectoryClient, 
                {'arm;vel' : self.teleop_commands['manipulator']['vel_extend']},
            )
            
            rospy.sleep(0.1)
            
        
    
        