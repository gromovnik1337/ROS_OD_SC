#ifndef PCL_LIDAR_VIEW_HPP
#define PCL_LIDAR_VIEW_HPP

// Include necessary libraries
//ROS
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
// PCL
#include <pcl/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/visualization/pcl_visualizer.h> 


class getAndView
{
private:
    // Create necessary ROS objects
    ros::NodeHandle _node_handle;

    // Create a subscriber
    ros::Subscriber _cloud_subscriber;

    // Create a visualizer object
    pcl::visualization::PCLVisualizer::Ptr _v;

public:
    getAndView();

    ~getAndView();

    void lidarRawDataCallback(const sensor_msgs::PointCloud2::ConstPtr& msg);
};

#endif 