#ifndef PCL_SCREENSHOT_HPP
#define PCL_SCREENSHOT_HPP

// Include necessary libraries
//ROS
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
// PCL
#include <pcl/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/visualization/pcl_visualizer.h> 
#include <pcl/io/pcd_io.h>


class getAndView
{
private:
    // Create necessary ROS objects
    ros::NodeHandle _node_handle;

    // Create a subscriber
    ros::Subscriber _cloud_subscriber;

public:
    getAndView();

    ~getAndView();

    void lidarRawDataCallback(const sensor_msgs::PointCloud2::ConstPtr& msg);
};

#endif 