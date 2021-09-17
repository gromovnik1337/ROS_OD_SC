// A program from AGITR book that waits for the turtlesim to start up,
// and then it changes its background colour.

#include <ros/ros.h>
#include <std_srvs/Empty.h>

int main(int argc, char **argv) {

    ros::init(argc, argv, "set_bg_color");
    ros::NodeHandle nh;

    // Wait until the clear service is available, which indicated that the turtle has booted up
    // and has set the background color parameters.

    ros::service::waitForService("clear");

    // Set the background color for the turtle - override the defaults
    ros::param::set("/turtlesim/background_r", 255);
    ros::param::set("/turtlesim/background_g", 255);
    ros::param::set("/turtlesim/background_b", 0);

    // Get the turtle to read the new values of the parameters
    ros::ServiceClient clearClient = nh.serviceClient<std_srvs::Empty>("/clear");
    std_srvs::Empty srv;
    clearClient.call(srv);

}     