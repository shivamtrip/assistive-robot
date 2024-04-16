# End-to-end autonomy (HRI, Perception, Planning, Navigation, Manipulation, Controls) on a Stretch RE1 robot

- This was my Master's capstone project at CMU Robotics. It involved developing and integrating an end-to-end autonomy stack for an indoor assistive robot.
- At the end of our project (Dec-2023), we successfully [deployed our robot at an Assisted Living Facility](https://www.youtube.com/watch?v=epFzxcuik8c&ab_channel=AuxilioRobotics) in Pittsburgh and Carnegie Mellon University [featured our work](https://www.cs.cmu.edu/news/2023/care-home-robot).

**Demo @ CMU Robotics Institute**
:-------------------------:|
<br /> <img src="https://github.com/shivamtrip/home-robot/assets/66013750/f39b4dbb-7791-4d4f-96a6-782c452834fd" width="450"> &nbsp; |

[<img src = "https://github.com/shivamtrip/assistive-robot/assets/66013750/42b94cf7-d9d8-4c68-a32f-f60633ff56ca" width = "400">](https://www.cs.cmu.edu/news/2023/care-home-robot) <br/> 

**Demo @ Vincentian Senior Living, Pittsburgh**
:-------------------------:|
&nbsp; <br />  <img src = "https://github.com/shivamtrip/assistive-robot/assets/66013750/8b7fc190-bc96-4c4f-8c32-a48c7bb1a066" width="450"> <br />


**Important Links:** [Project YouTube](https://www.youtube.com/@AuxilioRobotics), [Project Website](https://mrsdprojects.ri.cmu.edu/2023teamf), [Project Github](https://github.com/Auxilio-Robotics/alfred-deployed) 


## Use-Case: 
1. A user provides a voice command to Alfred (Stretch RE1 robot) for fetching an object.
2. Alfred understands this command and sets out to fetch the object.
   -  It first plans a path to the object's approximate location in the environment and then autonomously navigates to that location.
   -  Upon reaching the object location, Alfred searches for the object.
   -  Once it finds the object, it aligns with the object so it can successfully grasp it.
   -  Alfred then plans a path to grasping the object with its arm and goes on to grasp it.
   -  Once the object has been grasped, Alfred returns back to the user with the desired object.
3. The user is happy! :) 


## More Videos
Navigation             |  Manipulation
:-------------------------:|:-------------------------:
<img src="https://github.com/shivamtrip/assistive-robot/assets/66013750/1e100290-46ea-495f-a957-8b471560a2af" width="300"> &nbsp; | &nbsp; <img src = "https://github.com/shivamtrip/home-robot/assets/66013750/7ba8de89-31f0-4fcf-9dc2-8be52344d24c" width="250"> <br />
<img src = "https://github.com/shivamtrip/home-robot/assets/66013750/91732fd1-f02f-461a-99ab-a01f0a7eb123" width="300"> &nbsp;| &nbsp; <img src="https://github.com/shivamtrip/assistive-robot/assets/66013750/dd773590-5f12-440e-ad63-26ebbec67c77" width="250" > <br />

## System Architectures
#### Functional Architecture
<img src="https://github.com/shivamtrip/assistive-robot/assets/66013750/1073baaf-a9ab-43d1-9273-dcb43b4903da" width="600"> <br/>
#### Cyber-Physical Architecture 
<img src="https://github.com/shivamtrip/assistive-robot/assets/66013750/eb1b2101-ca88-47b5-a550-7b64e1747261" width="600"> <br/>

## Software Architectures
#### Overall Deployment Systen
<img src="https://github.com/shivamtrip/home-robot/assets/66013750/1de1b99d-f994-45a1-859a-482e0953d265" width="600"> <br/>

#### Mission Planner Node
<img src="https://github.com/shivamtrip/home-robot/assets/66013750/8a995bd9-c8f4-46ad-833a-9793f6b44d7d" width="600"> <br/>

#### Database Structure
Overall Structure |
:-------------------------:|
<img src="https://github.com/shivamtrip/assistive-robot/assets/66013750/f6991590-dfb7-49a9-999c-26b2972f46e2" width="500"> &nbsp;
  

## Tech Stack
- For HRI, we are using Google Cloud Speech-to-Text API and the ChatGPT API. 
- For Navigation, we are using ROS Movebase, STVL for 3D Obstacle Avoidance, GMapping for 2D Lidar SLAM, AMCL for 2D Localization, A* for global planning and DWA for local planning.
- For Perception and Manipulation, we are using Yolo v8 for object detection, Graspnet for grasp point detection and have implemented multiple finite state machines for different stages of our pipeline.
- For overall system integration, we have a system-level finite state machine.
- For low-level joint control, we are leveraging the Stretch RE1's open-source APIs.

**My primary role in this project:** 
- Use ROS / C++ / Python to set up and optimize the navigation stack (localization, planning & 3D obstacle avoidance)
- Design & implement a robust software architecture which integrates a cloud-server, HRI interfaces, navigation / manipulation stacks and a high-level task planner.
- Through the course of the project, I also contributed to various perception, manipulation and system-integration aspects of the project.



## Project Poster 
<img src = "https://github.com/shivamtrip/assistive-robot/assets/66013750/e754fc4b-5b5d-4be2-9317-6ea9829ebad7" width="600">


