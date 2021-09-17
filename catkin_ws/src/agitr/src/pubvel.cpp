// A program that publishes random generated velocity, to the turtlesim
// From A Gentle Introduction to ROS
// Created by: Vice, 21.03.2021

#include <pubvel.hpp>

// Constructor of the publishTurtle class
publishTurtle::publishTurtle()
{

// Create a publisher object
_turtle_velocity_publish = _node_handle.advertise<geometry_msgs::Twist>("turtle1/cmd_vel", 1000);

}

// De-cnstructor of the publishTurtle class
publishTurtle::~publishTurtle()
{

}

// Body of a function that call the publish method and writes out the output
void publishTurtle::writeVelocity(geometry_msgs::Twist msg)
{
    // Publish the message
    _turtle_velocity_publish.publish(msg);

    // Send the message to rosout with the details
    ROS_INFO_STREAM("Sending random velocity command:"
    << "linear=" << msg.linear.x
    << "angular=" << msg.angular.z);
}

int main(int argc, char **argv) 
{

    ros::init(argc, argv, "publish_velocity");
    ros::Time::init();

    // Seed the random number generator
    srand(time(0));

    // Declare a rate object - to control the speed of the loop
    ros::Rate rate(2);

    // Declare a message to publish turtle speed
    geometry_msgs::Twist msgToTurtle;

    // Instantiate the object of the class
    publishTurtle publishTurtleBasic;

    while(ros::ok()) 
{
    // Fill in the message. The other four fields, 
    // which are ignored by turtlesim, default to 0
    msgToTurtle.linear.x = double(rand())/double(RAND_MAX);
    msgToTurtle.angular.z = 2*double(rand()/double(RAND_MAX) - 1);

    publishTurtleBasic.writeVelocity(msgToTurtle);

    // Wait until it's time for another iteration.
    rate.sleep();
}
    ros::spin();

}




