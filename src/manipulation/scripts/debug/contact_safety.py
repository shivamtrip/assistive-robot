import numpy as np
import rospy
from helpers import move_to_pose
import threading
from std_srvs.srv import Trigger, TriggerResponse, TriggerRequest

class ContactDetector():
    def __init__(self, get_joint_state, in_contact_func, move_increment=0.008):
        self.in_contact_func = in_contact_func

        # reach until contact related
        self.in_contact = False
        self.in_contact_position = None
        self.contact_state_lock = threading.Lock()
        self.contact_mode = None
        self.contact_mode_lock = threading.Lock()
        
        self.position = None
        self.av_effort = None
        self.av_effort_window_size = 3

        self.get_joint_state = get_joint_state        
        self.direction_sign = None
        self.stopping_position = None

        self.min_av_effort_threshold = 10.0
        self.move_increment = move_increment

    def set_regulate_contact(self):
        with self.contact_mode_lock:
            return self.contact_mode == 'regulate_contact'
        
    def set_stopping_position(self, stopping_position, direction_sign):
        assert((direction_sign == -1) or (direction_sign == 1))
        self.stopping_position = stopping_position
        self.direction_sign = direction_sign

    def is_in_contact(self):
        with self.contact_state_lock:
            return self.in_contact

    def contact_position(self):
        with self.contact_state_lock:
            return self.in_contact_position
         
    def get_position(self):
        return self.position
    
    def passed_stopping_position(self):
        if (self.position is None) or (self.stopping_position is None):
            return False
        difference = self.stopping_position - self.position
        if int(np.sign(difference)) == self.direction_sign:
            return False
        return True

    def not_stopped(self):
        with self.contact_mode_lock:
            return self.contact_mode == 'stop_on_contact'
    
    def reset(self): 
        with self.contact_state_lock:
            self.in_contact = False
            self.in_contact_position = None
        self.turn_off()
        self.stopping_position = None
        self.direction_sign = None
        
    def turn_off(self):
        with self.contact_mode_lock:
            self.contact_mode = None

    def turn_on(self):
        with self.contact_mode_lock:
            self.contact_mode = 'stop_on_contact'

    def update(self, joint_states, stop_the_robot_service):
        with self.contact_state_lock:
            self.in_contact = False
            self.in_contact_wrist_position = None

        position, velocity, effort = self.get_joint_state(joint_states)
        self.position = position

        # First, check that the stopping position, if defined, has not been passed
        if self.passed_stopping_position(): 
            trigger_request = TriggerRequest()
            trigger_result = stop_the_robot_service(trigger_request)
            with self.contact_mode_lock:
                self.contact_mode = 'passed_stopping_point'
            rospy.loginfo('stop_on_contact: stopping the robot due to passing the stopping position, position = {0}, stopping_position = {1}, direction_sign = {2}'.format(self.position, self.stopping_position, self.direction_sign))

        # Second, check that the effort thresholds have not been exceeded
        if self.av_effort is None:
            self.av_effort = effort
        else:
            self.av_effort = (((self.av_effort_window_size - 1.0) * self.av_effort) + effort) / self.av_effort_window_size

        if self.in_contact_func(effort, self.av_effort):
            # Contact detected!
            with self.contact_state_lock:
                self.in_contact = True
                self.in_contact_position = self.position
            with self.contact_mode_lock:
                if self.contact_mode == 'stop_on_contact':
                    trigger_request = TriggerRequest()
                    trigger_result = stop_the_robot_service(trigger_request)
                    rospy.loginfo('stop_on_contact: stopping the robot due to detected contact, effort = {0}, av_effort = {1}'.format(effort, self.av_effort))
                    self.contact_mode = 'regulate_contact'
                elif self.contact_mode == 'regulate_contact':
                    pass
        elif self.av_effort < self.min_av_effort_threshold:
            with self.contact_mode_lock:
                if self.contact_mode == 'regulate_contact':
                    pass
        else:
            pass

            
    def move_until_contact(self, joint_name, stopping_position, direction_sign, move_to_pose):
        self.reset()
        self.set_stopping_position(stopping_position, direction_sign)
        
        success = False
        message = 'Unknown result.'
        
        if not self.passed_stopping_position():
            # The target has not been passed
            self.turn_on()

            move_rate = rospy.Rate(5.0)
            move_increment = direction_sign * self.move_increment
            finished = False
            while self.not_stopped():
                position = self.get_position()
                if position is not None: 
                    new_target = self.get_position() + move_increment
                    pose = {joint_name : new_target}
                    move_to_pose(pose)
                move_rate.sleep()

            if self.is_in_contact():
                # back off from the detected contact location
                
                contact_position = self.contact_position()
                if contact_position is not None: 
                    new_target = contact_position - 0.001 #- 0.002
                else:
                    new_target = self.position() - 0.001 #- 0.002
                pose = {joint_name : new_target}
                move_to_pose(pose)
                rospy.loginfo('backing off after contact: moving away from surface to decrease force')
                success = True
                message = 'Successfully reached until contact.'
            else: 
                success = False
                message = 'Terminated without detecting contact.'

        self.reset()

        return success, message
