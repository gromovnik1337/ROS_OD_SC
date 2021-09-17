// Hello world program for ROS
// Created by: Vice, 12.03.2021

// Header with standard ROS classes
#include <ros/ros.h>

int main(int argc, char **argv) {
    // Initialize ROS system
    ros::init(argc, argv, "hello_ros");

    // Establish this program as a ROS node, creates an NodeHandle object
    ros::NodeHandle nh;

    //Send an output
    ROS_INFO_STREAM("Hello, ROS!");
}