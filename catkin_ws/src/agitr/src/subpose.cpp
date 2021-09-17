// A program that subscribes tp the turtle1/pose and outputs its messages to the screen
// From A Gentle Introduction to ROS
// Created by: Vice, 21.03.2021

#include <subpose.hpp>

// Constructor of the subscribeTurtle class
subscribeTurtle::subscribeTurtle()
{

    // Create a subscriber object
    _turtle_pose_subscribe = _node_handle.subscribe("turtle1/pose", 1000, &subscribeTurtle::poseMessageReceived, this); 

}

// De-constructor of the subscribeTurtle class
subscribeTurtle::~subscribeTurtle()
{

}

// Body of a publisher callback function
void subscribeTurtle::poseMessageReceived(const turtlesim::Pose& msg)

{
    ROS_INFO_STREAM(std::setprecision(2) << std::fixed << "postion=(" << msg.x << "." << msg.y << ")" << "direction=" << msg.theta);
}

int main (int argc, char **argv)
{
    ros::init(argc, argv, "subcribe_to_pose");
    ros::Time::init();

    subscribeTurtle subscribeToTurtleBasic;

    ros::spin();

}

