// Livox Mid-40 data acquisition program
// Created by: Vice, 15.03.2021

#include <acquisition.hpp>

// Constructor of the acquisition class
acquisition::acquisition()
{

    _cloud_subscriber = _node_handle.subscribe<sensor_msgs::PointCloud2>("/livox/lidar", 1, &acquisition::topicCallBack, this);
}

// Deconstructor of the acquisition class
acquisition::~acquisition()
{
}

void acquisition::topicCallBack(const sensor_msgs::PointCloud2::ConstPtr& msg)
{
   
    // Transform the message to a pcl::PCLPointCloud2
    pcl::PCLPointCloud2 pointCloud2;
    pcl_conversions::toPCL(*msg, pointCloud2);

    ROS_INFO_STREAM("Data acquired");
}

int main(int argc, char **argv) {

    ros::init(argc, argv, "acquisition");
    ros::Time::init();

    // Announce the start of the process
    ROS_INFO_STREAM("Data acquisition initiated!");

    // Initialize
    acquisition acquireBasic;

    ros::spin();

}