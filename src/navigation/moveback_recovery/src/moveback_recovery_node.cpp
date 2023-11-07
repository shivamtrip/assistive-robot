#include "ros/ros.h"
#include <moveback_recovery/moveback_recovery.h>
#include <tf2_ros/transform_listener.h>
#include <actionlib/server/simple_action_server.h>
#include <moveback_recovery/MovebackRecoveryAction.h>

class MovebackRecoveryAction{

protected:

  ros::NodeHandle nh_;
  actionlib::SimpleActionServer<moveback_recovery::MovebackRecoveryAction> as_; // NodeHandle instance must be created before this line. Otherwise strange error occurs.
  std::string action_name_;
  // create messages that are used to published feedback/result
  moveback_recovery::MovebackRecoveryFeedback feedback_;
  moveback_recovery::MovebackRecoveryResult result_;


public:

  MovebackRecoveryAction(std::string name) :
    as_(nh_, name, boost::bind(&MovebackRecoveryAction::executeCB, this, _1), false),
    action_name_(name)
  {
    as_.start();
  }

  ~MovebackRecoveryAction(void)
  {
  }

  void executeCB(const moveback_recovery::MovebackRecoveryGoalConstPtr &goal)
  {
    // moveback_object -> runBehavior();
  }


};



int main(int argc, char** argv){

  costmap_2d::Costmap2DROS* planner_costmap_ros_;
  costmap_2d::Costmap2DROS* controller_costmap_ros_;

  ros::init(argc, argv, "moveback_recovery");

  MovebackRecoveryAction moveback_recovery("moveback_recovery");

  tf2_ros::Buffer buffer(ros::Duration(10));
  tf2_ros::TransformListener tf(buffer);

  //create the ros wrapper for the planner's costmap... and initializer a pointer we'll use with the underlying map
  planner_costmap_ros_ = new costmap_2d::Costmap2DROS("global_costmap", buffer);
  planner_costmap_ros_->pause();

  //create the ros wrapper for the controller's costmap... and initializer a pointer we'll use with the underlying map
  controller_costmap_ros_ = new costmap_2d::Costmap2DROS("local_costmap", buffer);
  controller_costmap_ros_->pause();

  // Start actively updating costmaps based on sensor data
  planner_costmap_ros_->start();
  controller_costmap_ros_->start();

  moveback_recovery::MovebackRecovery moveback_object();
  // moveback_object.initialize("moveback_recovery", &buffer, planner_costmap_ros_, controller_costmap_ros_);

  ros::spin();

  return(0);
}