#ifndef SUBPOSE_hpp
#define SUBPOSE_hpp

// Header with standard ROS classes
#include <ros/ros.h>

// Turtle pose header, recquired for the message type
#include <turtlesim/Pose.h>

// Auxiliary header, used for std::setprecision and std::fixed
#include <iomanip>

class subscribeTurtle
{
private:

    // Declare necessary ROS objects
    ros::NodeHandle _node_handle;

    // Declare a subscriber object
    ros::Subscriber _turtle_pose_subscribe;

public:

    subscribeTurtle();

    ~subscribeTurtle();

    void poseMessageReceived(const turtlesim::Pose& msg);
};

#endif