# End-to-end autonomy (HRI, Perception, Planning, Navigation, Manipulation, Controls) on a Stretch RE1 robot

- This was my Master's capstone project at CMU Robotics. It involved developing and integrating an end-to-end autonomy stack for an indoor assistive robot.
- At the end of our project (Dec-2023), we successfully [deployed our robot at an Assisted Living Facility](https://www.youtube.com/watch?v=epFzxcuik8c&ab_channel=AuxilioRobotics) in Pittsburgh and Carnegie Mellon University [featured our work](https://www.cs.cmu.edu/news/2023/care-home-robot).

<img src = "https://github.com/shivamtrip/home-robot/assets/66013750/f39b4dbb-7791-4d4f-96a6-782c452834fd" width="450"> <br/>

**Important Links:** [Project YouTube](https://www.youtube.com/@AuxilioRobotics), [Project Website](https://mrsdprojects.ri.cmu.edu/2023teamf), [Project Github](https://github.com/Auxilio-Robotics/alfred-deployed) 

## Implementation
The user provides a voice command to Alfred (Stretch RE1 robot) for fetching an object. Alfred understands this command, plans and navigates to the object's approximate location in the environment, detects and grasps the object and finally returns back to the user with the desired object. 

- For HRI, we are using Google Cloud Speech-to-Text API and the ChatGPT API. 
- For Navigation, we are using ROS Movebase, STVL for 3D Obstacle Avoidance, GMapping for 2D Lidar SLAM, AMCL for 2D Localization, A* for global planning and DWA for local planning.
- For Perception and Manipulation, we are using Yolo v8, Graspnet and have implemented multiple finite state machines.
- For overall system integration, we have a system-level finite state machine.
- For low-level joint control, we are leveraging the Stretch RE1's open-source APIs.

**My primary role in this project:** 
- Use ROS / C++ / Python to set up and optimize the navigation stack (localization, planning & 3D obstacle avoidance)
- Design & implement a robust software architecture which integrates a cloud-server, HRI interfaces, navigation / manipulation stacks and a high-level task planner.
- Through the course of the project, I also contributed to various perception, manipulation and system-integration aspects of the project.


## Videos
Navigation             |  Manipulation
:-------------------------:|:-------------------------:
<img src="https://github.com/shivamtrip/assistive-robot/assets/66013750/1e100290-46ea-495f-a957-8b471560a2af" width="300"> &nbsp; | &nbsp; <img src = "https://github.com/shivamtrip/home-robot/assets/66013750/7ba8de89-31f0-4fcf-9dc2-8be52344d24c" width="250"> <br />
<img src = "https://github.com/shivamtrip/home-robot/assets/66013750/91732fd1-f02f-461a-99ab-a01f0a7eb123" width="300"> &nbsp;| &nbsp; <img src="https://github.com/shivamtrip/assistive-robot/assets/66013750/dd773590-5f12-440e-ad63-26ebbec67c77" width="250" > <br />

## System Architectures
<img src="https://github.com/shivamtrip/assistive-robot/assets/66013750/1073baaf-a9ab-43d1-9273-dcb43b4903da" width="600"> <br/>
<img src="https://github.com/shivamtrip/assistive-robot/assets/66013750/eb1b2101-ca88-47b5-a550-7b64e1747261" width="600"> <br/>

## Software Architectures
<img src="https://github.com/shivamtrip/home-robot/assets/66013750/1de1b99d-f994-45a1-859a-482e0953d265" width="600"> <br/>
<img src="https://github.com/shivamtrip/home-robot/assets/66013750/8a995bd9-c8f4-46ad-833a-9793f6b44d7d" width="600"> <br/>




