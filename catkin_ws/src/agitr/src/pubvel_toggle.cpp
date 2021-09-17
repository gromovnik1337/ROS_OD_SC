// This program toggles between rotation and translation commands for the turtle.
// It does that each time a service is called which is registered inside the program.
// To test this program, start the turtlesim_node and run this node. Then call the service inside this node with rosservice call /toggle_forward

#include <ros/ros.h>
#include <std_srvs/Empty.h>
#include <geometry_msgs/Twist.h>

bool forward = true;
bool toggleForward(std_srvs::Empty::Request &req, std_srvs::Empty::Response &resp) {
    // Both of the arguments are in this case empty so there is no processing required.
    // This callback onl toggles global boolean variable "forward".
    forward = !forward;
    ROS_INFO_STREAM("Now sending" << (forward ? " forward":" rotate")<< " commands.");

    return true;
}

int main (int argc, char **argv){
    ros::init(argc, argv, "pubvel_toggle");
    ros::NodeHandle nh;

    // Register the service with the master.
    ros::ServiceServer server = nh.advertiseService("toggle_forward", &toggleForward);

    // Publish commands, using the latest value for forward, until the node shuts down.
    ros::Publisher pub = nh.advertise<geometry_msgs::Twist>("turtle1/cmd_vel", 1000);
    ros::Rate rate(2);
    while(ros::ok()) {
        geometry_msgs::Twist msg;
        msg.linear.x = forward ? 1.0 : 0.0;
        msg.angular.z = forward ? 0.0 : 1.0;
        pub.publish(msg);
        ros::spinOnce(); // Used because velocity commands are published when there are no incoming service calls to process. 
        rate.sleep();
    }
}
