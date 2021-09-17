#ifndef REVERSE_hpp
#define REVERSE_hpp

// Header with standard ROS classes
#include <ros/ros.h>
// Message type recognized by the turtle
#include <geometry_msgs/Twist.h>

class publishReverse
{

private:

    // Declare necessary ROS objects
    ros::NodeHandle _node_handle;

    // Declare a publisher object
    ros::Publisher *_pubPtr;

public:

    publishReverse();

    ~publishReverse();

    void commandVelocityReceived(const geometry_msgs::Twist&);

};

#endif