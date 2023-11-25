#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/passthrough.h>

float fx, fy, cx, cy;
bool intrinsics_set = false;
ros::Publisher point_cloud_pub;
bool start_pub = false;
int scale_factor = 3;
void convertDepthImageToPointcloud(const sensor_msgs::ImageConstPtr &depth_msg, pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud)
{
  // Assuming 16-bit unsigned short depth image
  cv::Mat depth_image = cv_bridge::toCvCopy(depth_msg, sensor_msgs::image_encodings::TYPE_16UC1)->image;
  // Create point cloud
  float depth_factor = 1000.0;
  cloud->width = (int)(depth_image.cols / scale_factor);
  cloud->height = (int) (depth_image.rows / scale_factor);
  cloud->is_dense = false;
  cloud->points.resize((cloud->width * cloud->height));
  if(!start_pub){
    start_pub = true;
    ROS_INFO("Started publishing point cloud...");
    ROS_INFO("CLOUD SIZE: %d", cloud->points.size());
  }
  int i = 0;
  for (int y = 0; y < depth_image.rows; y += scale_factor)
  {
    for (int x = 0; x < depth_image.cols; x += scale_factor, ++i)
    {
      // ROS_INFO("FIlling: %d", i);
      if(i >= cloud->width * cloud->height) {
        // ROS_INFO("i: %d, width: %d, height: %d", i, cloud->width, cloud->height);
        
        break;
      }
      pcl::PointXYZ &point = cloud->points[i];
      uint16_t depth_value = depth_image.at<uint16_t>(y, x);

      // Check if the depth value is valid
      if (depth_value == 0 || depth_value == 65535)
      {
        point.x = point.y = point.z = std::numeric_limits<float>::quiet_NaN();
      }
      else
      {
        point.z = depth_value / depth_factor;
        point.x = (x - cx) * point.z / fx;
        point.y = (y - cy) * point.z / fy;
      }
    }
  }
}

void imageCallback(const sensor_msgs::ImageConstPtr &msg)
{
  try
  {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    

    convertDepthImageToPointcloud(msg, cloud);
    sensor_msgs::PointCloud2 pc_msg;
    pcl::toROSMsg(*cloud, pc_msg);
    pc_msg.header = msg->header;
    pc_msg.header.frame_id = "camera_color_optical_frame";
    point_cloud_pub.publish(pc_msg);
  }
  catch (cv_bridge::Exception &e)
  {
    ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
  }
}
void infoCallback(const sensor_msgs::CameraInfoConstPtr &info_msg)
{
  // Store camera intrinsics
  if (!intrinsics_set)
  {
    fx = info_msg->K[0]; // Focal length x
    fy = info_msg->K[4]; // Focal length y
    cx = info_msg->K[2]; // Principal point x
    cy = info_msg->K[5]; // Principal point y
    ROS_INFO("fx: %d, fy: %d, cx: %d, cy: %d", fx, fy, cx, cy);
    intrinsics_set = true;
  }
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "pcd_pub");
  ros::NodeHandle nh;
  image_transport::ImageTransport it(nh);
  ros::Subscriber info_sub = nh.subscribe("/camera/color/camera_info", 1, &infoCallback);
  scale_factor = ros::param::param<int>("/pcd_pub/downscale_factor", 3);
  while (!intrinsics_set)
  {
    ros::spinOnce();
  }

  image_transport::Subscriber sub = it.subscribe("/camera/aligned_depth_to_color/image_raw", 1, imageCallback);
  point_cloud_pub = nh.advertise<sensor_msgs::PointCloud2>("/low_res_pointcloud", 1);
  ros::spin();
}
