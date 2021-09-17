// This program subscribes to turtle1/cmd_vel and publishes to
// turtle1/cmd_vel_reversed with the inverted message content
// Created by: Vice, 06.04.2021

#include <reverse_cmd_vel.hpp>

// Constructor of the publishReverse class
publishReverse::publishReverse()
{

    // Create a publisher object
    _pubPtr = new ros::Publisher(_node_handle.advertise<geometry_msgs::Twist>("turtle1/cmd_vel_reversed", 1000));

    // Create a subscriber object
    ros::Subscriber sub = _node_handle.subscribe("/turtle1/cmd_vel", 1000, &publishReverse::commandVelocityReceived, this);

}

// Deconstructor of the publishReverse class
publishReverse::~publishReverse()
{ 

}

void publishReverse::commandVelocityReceived(const geometry_msgs::Twist& msgIn)
{
    geometry_msgs::Twist msgOut;
    msgOut.linear.x = -msgOut.linear.x;
    msgOut.linear.z = -msgOut.linear.z;

    _pubPtr->publish(msgOut);
}

int main (int argc, char **argv)
{
    ros::init(argc, argv, "reverse_velocity");

    // Instantiate an object of publishReverse class
    publishReverse publishReverseBasic;

    ros::spin();

    //delete _pubPtr;

}

