#ifndef ACQUISITION_HPP
#define ACQUISITION_HPP

// Include necessary libraries
//ROS
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
// PCL
#include <pcl/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h> 



class acquisition
{
private:
    // Create necessary ROS objects
    ros::NodeHandle _node_handle;

    // Create a subscriber
    ros::Subscriber _cloud_subscriber;

public:
    acquisition();

    ~acquisition();

    void topicCallBack(const sensor_msgs::PointCloud2::ConstPtr& msg);

};

#endif 