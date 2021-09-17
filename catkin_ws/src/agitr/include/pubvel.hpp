#ifndef PUBVEL_hpp
#define PUBVEL_hpp

// Header with standard ROS classes
#include <ros/ros.h>
// Message type recognized by the turtle
#include <geometry_msgs/Twist.h>
// Recquired for rand() and RAND_MAX
#include <stdlib.h> 

class publishTurtle 
{
private:
    // Declare necessary ROS objects
    ros::NodeHandle _node_handle;

    // Declare a publisher object
    ros::Publisher _turtle_velocity_publish;

public:
    publishTurtle();

    ~publishTurtle();

    void writeVelocity(geometry_msgs::Twist);

}; 

#endif