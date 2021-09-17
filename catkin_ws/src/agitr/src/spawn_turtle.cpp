// This program spawns a new turtlesim turtle by calling the appropriate ros service.
#include <ros/ros.h>
// srv class for the service.
#include <turtlesim/Spawn.h>

int main(int argc, char **argv) {
    ros::init(argc, argv, "spawn_turtle");
    ros::NodeHandle nh;

    // Create a client object which takes the data type of the service and its name as an argument.
    ros::ServiceClient spawnClient = nh.serviceClient<turtlesim::Spawn>("spawn");

    // Create the request and response objects.
    turtlesim::Spawn::Request req;
    turtlesim::Spawn::Response resp;

    // Fill in the request object data members. Response object data members are not filled because those informations are provided by the server.
    req.x = 2;
    req.y = 3;
    req.theta = M_PI / 2;
    req.name = "Leo";

    // Call the service. This call will return nothing until the service is complete.
    bool success = spawnClient.call(req, resp);

    // Check for success and use the response.
    if(success) {
        ROS_INFO_STREAM("Spawned a turtle named " << resp.name);
    } else {
        ROS_ERROR_STREAM("Failed to spawn.");

    }

}