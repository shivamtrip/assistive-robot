#include <moveback_recovery/moveback_recovery.h>
#include <pluginlib/class_list_macros.h>
#include <nav_core/parameter_magic.h>
#include <tf2/utils.h>
#include <ros/ros.h>
#include <geometry_msgs/Twist.h>
#include <geometry_msgs/Point.h>
#include <angles/angles.h>
#include <algorithm>
#include <string>
#include <tf2_ros/transform_listener.h>
#include <geometry_msgs/TransformStamped.h>


// register this planner as a RecoveryBehavior plugin
PLUGINLIB_EXPORT_CLASS(moveback_recovery::MovebackRecovery, nav_core::RecoveryBehavior)

namespace moveback_recovery
{
MovebackRecovery::MovebackRecovery(): local_costmap_(NULL), initialized_(false), world_model_(NULL)
{
}

void MovebackRecovery::initialize(std::string name, tf2_ros::Buffer*,
                                costmap_2d::Costmap2DROS*, costmap_2d::Costmap2DROS* local_costmap)
{
  if (!initialized_)
  {
    local_costmap_ = local_costmap;

    // get some parameters from the parameter server
    ros::NodeHandle private_nh("~/" + name);
    ros::NodeHandle blp_nh("~/TrajectoryPlannerROS");

    // we'll simulate every degree by default
    private_nh.param("sim_granularity", sim_granularity_, 0.017);
    private_nh.param("frequency", frequency_, 20.0);

    acc_lim_th_ = nav_core::loadParameterWithDeprecation(blp_nh, "acc_lim_theta", "acc_lim_th", 3.2);
    max_rotational_vel_ = nav_core::loadParameterWithDeprecation(blp_nh, "max_vel_theta", "max_rotational_vel", 1.0);
    min_rotational_vel_ = nav_core::loadParameterWithDeprecation(blp_nh, "min_in_place_vel_theta", "min_in_place_rotational_vel", 0.4);
    blp_nh.param("yaw_goal_tolerance", tolerance_, 0.10);

    world_model_ = new base_local_planner::CostmapModel(*local_costmap_->getCostmap());

    initialized_ = true;
  }
  else
  {
    ROS_ERROR("You should not call initialize twice on this object, doing nothing");
  }
}

MovebackRecovery::~MovebackRecovery()
{
  delete world_model_;
}

void MovebackRecovery::runBehavior()
{
  if (!initialized_)
  {
    ROS_ERROR("This object must be initialized before runBehavior is called");
    return;
  }

  if (local_costmap_ == NULL)
  {
    ROS_ERROR("The costmap passed to the MovebackRecovery object cannot be NULL. Doing nothing.");
    return;
  }
  ROS_WARN("Starting Moveback Recovery.");

  ros::Rate r(frequency_);
  ros::NodeHandle n;
  ros::Publisher vel_pub = n.advertise<geometry_msgs::Twist>("cmd_vel", 10);

  geometry_msgs::PoseStamped global_pose;
  local_costmap_->getRobotPose(global_pose);


  bool reverse = true;      // if reverse is true, move back


  tf2_ros::Buffer tfBuffer;
  tf2_ros::TransformListener tfListener(tfBuffer);

  geometry_msgs::TransformStamped transformStamped;

  while (n.ok()){
    try{
      transformStamped = tfBuffer.lookupTransform("base_link", "map",
                               ros::Time(0));
      break;
    }
    catch (tf2::TransformException &ex) {
      ROS_WARN("%s",ex.what());
      ros::Duration(1.0).sleep();
      continue;
    }
  }

  double init_x = transformStamped.transform.translation.x;
  double init_y = transformStamped.transform.translation.y;

  double global_x_init = global_pose.pose.position.x;       //Position in Global Map Coordinates
  double global_y_init = global_pose.pose.position.y;       //Position in Global Map Coordinates
  
  std::cout << "Cur Position: " << init_x << " " << init_y <<  std::endl;
  std::cout << "Global: " << global_x_init << " " << global_y_init <<  std::endl;

  ros::Time begin_time =ros::Time::now();

  double cur_x; 
  double cur_y; 
  double dist_travelled = 0; 

  cur_x = init_x;
  cur_y = init_y;
    
  while (n.ok() && reverse)
  {

    ros::Time cur_time = ros::Time::now();

    ros::Duration time_diff = cur_time - begin_time;
    ros::Duration timeout_duration(6, 0);

    if (time_diff > timeout_duration){
      std::cout << "Exiting Moveback Recovery due to 6 second timeout" << std::endl;
      break;
    }

    try{
      transformStamped = tfBuffer.lookupTransform("base_link", "map",
                               ros::Time(0));
    }
    catch (tf2::TransformException &ex) {
      ROS_WARN("%s",ex.what());
      ros::Duration(1.0).sleep();
      continue;
    }

    double max_reverse_dist = 0.3;
    double reverse_dist = 0.1;
    double cur_yaw = tf2::getYaw(global_pose.pose.orientation);

    double global_x_cur = global_x_init - (dist_travelled * cos(cur_yaw));    //Cur position in Global Map
    double global_y_cur = global_y_init - (dist_travelled * sin(cur_yaw));    //Cur position in Global Map

    double sim_x = global_x_cur - (reverse_dist * cos(cur_yaw));
    double sim_y = global_y_cur - (reverse_dist * sin(cur_yaw));

    // make sure that the point is legal, if it isn't... we'll abort
    double footprint_cost = world_model_->footprintCost(sim_x, sim_y, cur_yaw, local_costmap_->getRobotFootprint(), 0.0, 0.0);
    if (footprint_cost < 0.0)
    {
      ROS_ERROR("Moveback recovery can't reverse because there is a potential collision. Cost: %.2f",
                footprint_cost);
      break;
    }

    cur_x = transformStamped.transform.translation.x;
    cur_y = transformStamped.transform.translation.y;
    double x_diff = cur_x - init_x;
    double y_diff = cur_y - init_y;
    dist_travelled = sqrt(x_diff * x_diff + y_diff * y_diff);

    if (dist_travelled > max_reverse_dist){
      reverse = false;            // If the robot has moved 0.5m backwards, stop reversing
    }
    
    std::cout << "Distance: " << dist_travelled << std::endl;

    geometry_msgs::Twist cmd_vel;
    cmd_vel.linear.x = -0.1;
    cmd_vel.linear.y = 0.0;
    cmd_vel.angular.z = 0.0; //vel

    vel_pub.publish(cmd_vel);

    r.sleep();
  }


  // Rotate Recovery Behavior

  bool done_rotating = false;
  double current_angle = tf2::getYaw(global_pose.pose.orientation);
  double start_angle = current_angle;
  double rotation_angle = 45;
  double goal_angle = start_angle + rotation_angle * (M_PI/180);
  double dist_left = rotation_angle;

  std::cout << "Rotating In-Place " << dist_travelled << std::endl;
  
  while (n.ok() &&
        (!done_rotating))
  {
    // Update Current Angle
    local_costmap_->getRobotPose(global_pose);
    current_angle = tf2::getYaw(global_pose.pose.orientation);
 
    // compute the distance left to rotate
    if (!done_rotating)
    {

      std::cout << "The distance left to rotate is " << dist_left * (180/M_PI) << " degrees" << std::endl;

      double distance_to_goal = std::fabs(angles::shortest_angular_distance(current_angle, goal_angle));
      dist_left = distance_to_goal;

      if (distance_to_goal < tolerance_)
      {
        std::cout << "Done rotating!" << std::endl;
        done_rotating = true;
      }

      
    }
    // else
    // {
    //   // If we have hit the 180, we just have the distance back to the start
    //   dist_left = std::fabs(angles::shortest_angular_distance(current_angle, start_angle));
    // }

    double x = global_pose.pose.position.x, y = global_pose.pose.position.y;

    // check if that velocity is legal by forward simulating
    double sim_angle = 0.0;
    while (sim_angle < dist_left)
    {
      double theta = current_angle + sim_angle;

      // make sure that the point is legal, if it isn't... we'll abort
      double footprint_cost = world_model_->footprintCost(x, y, theta, local_costmap_->getRobotFootprint(), 0.0, 0.0);
      if (footprint_cost < 0.0)
      {
        ROS_ERROR("Rotate recovery can't rotate in place because there is a potential collision. Cost: %.2f",
                  footprint_cost);
        return;
      }

      sim_angle += sim_granularity_;
    }

    // compute the velocity that will let us stop by the time we reach the goal
    double vel = sqrt(2 * acc_lim_th_ * dist_left);

    // make sure that this velocity falls within the specified limits
    vel = std::min(std::max(vel, min_rotational_vel_), max_rotational_vel_);

    geometry_msgs::Twist cmd_vel;
    cmd_vel.linear.x = 0.0;
    cmd_vel.linear.y = 0.0;
    cmd_vel.angular.z = vel;

    vel_pub.publish(cmd_vel);

    r.sleep();
  }
}
};  // namespace moveback_recovery