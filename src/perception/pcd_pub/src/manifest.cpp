#include <pluginlib/class_list_macros.hpp>
#include <pcd_pub/resized_publisher.h>
#include <pcd_pub/resized_subscriber.h>

PLUGINLIB_EXPORT_CLASS(ResizedPublisher, image_transport::PublisherPlugin)

PLUGINLIB_EXPORT_CLASS(ResizedSubscriber, image_transport::SubscriberPlugin)
