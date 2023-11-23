#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/passthrough.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

class DepthToPointCloudConverter
{
public:
    DepthToPointCloudConverter()
    {
        // Initialize ROS node handle
        nh_ = ros::NodeHandle("pcd_publisher");
        // Subscribe to the depth image topic
        depth_image_sub_ = nh_.subscribe("/camera/aligned_color_to_depth/image_raw", 1, &DepthToPointCloudConverter::depthImageCallback, this);
        info_sub = nh_.subscribe("/camera/aligned_color_to_depth/camera_info", 1, &DepthToPointCloudNode::infoCallback, this);
        // Advertise the point cloud topic
        point_cloud_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("/low_res_pointcloud", 1);
    }

    void depthImageCallback(const sensor_msgs::ImageConstPtr &depth_msg)
    {
        // Convert depth image to point cloud
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        convertDepthImageToPointcloud(depth_msg, cloud);

        sensor_msgs::PointCloud2 pc_msg;
        pcl::toROSMsg(*cloud, pc_msg);
        pc_msg.header = depth_msg->header;
        point_cloud_pub_.publish(pc_msg);
    }
    void infoCallback(const sensor_msgs::Ca &info_msg)
    {
        // Store camera intrinsics
        if (!intrinsics_set)
        {
            fx = info_msg.K[0]; // Focal length x
            fy = info_msg.K[4]; // Focal length y
            cx = info_msg.K[2]; // Principal point x
            cy = info_msg.K[5]; // Principal point y
            intrinsics_set = true;
        }
    }
    void convertDepthImageToPointcloud(const sensor_msgs::ImageConstPtr &depth_msg, pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud)
    {
        // Assuming 16-bit unsigned short depth image
        cv::Mat depth_image = cv_bridge::toCvCopy(depth_msg, sensor_msgs::image_encodings::TYPE_16UC1)->image;

        // Create point cloud
        cloud->width = depth_image.cols;
        cloud->height = depth_image.rows;
        cloud->is_dense = false;

        size_t size = cloud->width * cloud->height / (scale_factor * scale_factor);
        cloud->points.resize(size);

        

        int i = 0;
        for (int v = 0; v < depth_image.rows; v += scale_factor)
        {
            for (int u = 0; u < depth_image.cols; u += scale_factor, ++i)
            {
                pcl::PointXYZ &point = cloud->points[i];
                uint16_t depth_value = depth_image.at<uint16_t>(v, u);

                // Check if the depth value is valid
                if (depth_value == 0 || depth_value == 65535)
                {
                    point.x = point.y = point.z = std::numeric_limits<float>::quiet_NaN();
                }
                else
                {
                    point.z = depth_value / factor;
                    point.x = (u - cx) * point.z / fx;
                    point.y = (v - cy) * point.z / fy;
                }
            }
        }
    }

private:
    ros::NodeHandle nh_;
    ros::Subscriber depth_image_sub_;
    ros::Publisher point_cloud_pub_;
    ros::Publisher info_sub;
    int fx, fy, cx, cy;
    bool intrinsics_set = false;
    int scale_factor = 3;
    float factor = 1000.0; // Depth image factor
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "depth_to_pointcloud_node");

    DepthToPointCloudConverter converter;

    ros::spin();

    return 0;
}
