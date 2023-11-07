#!/usr/local/lib/robot_env/bin/python3

from collections import deque 
import rospy
from alfred_msgs.msg import DeploymentTask

# class States(Enum):
    
#     IDLE = 1


# class Task_Message:
    
#     def __init__(self):
#         self.


class Mission_Planner():
    
    def __init__(self):
        
        rospy.init_node("mission_planner")

        self.task_queue = deque()
        
        # self.current_task_type = 
        # self.current_task_object = 
        # self.current_task_room = 
        # self.current_goal_x = 
        # self.current_goal_y =         
        
        
        self.task_listener = rospy.Subscriber('deployment_task_info', DeploymentTask, self.task_allocator)        # callback: task_allocator
        
        # health_client()             # callback: 
        
        
        
    def task_allocator(self, msg):
        """
        Task Allocator adds information regarding the newly added task to the task queue    
        """
        
        # self.task_queue.append()
        
        print("Just received a new task from the Interface Manager!")
        print(msg)
    
    
    def health_manager():
        """
        Health Manager updates battery status (and other health parameters, if applicable) 
        """
    
            
    # def update_next_task():
        
    #     next_task = task_queue
        
    #     self.current_task_type = 
    #     self.current_task_object = 
    #     self.current_task_room = 
    #     self.current_goal_x = 
    #     self.current_goal_y = 
        
                    
            
    # def main():

    #     if not self.battery_health_ok or not self.task_queue or system_state != STATES.IDLE:
    #         print("Cannot perform next task")
    
    #     if self.task_queue:
    #         self.update_next_task()         # Updates all relevant parameters to perform next task



if __name__ == "__main__":
    
    mission_planner = Mission_Planner()
    
    
    while not rospy.is_shutdown():
        rospy.spin()

    print("Shutting down")
    
    