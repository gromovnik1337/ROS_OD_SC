// This script acquires the Livox Mid 40 data and subsequently visualizes them inside the PCL viewer
// Created by: Vice, 22.04.2021

#include <pcl_lidar_view.hpp>

// Declaration of the pointer for the visualizer is necessary outside the constructor of the class
getAndView::getAndView():
_v (new pcl::visualization::PCLVisualizer("Viewer"))
{

    _cloud_subscriber = _node_handle.subscribe<sensor_msgs::PointCloud2>("/livox/lidar", 1, &getAndView::lidarRawDataCallback, this);

    _v->setBackgroundColor (0, 0, 0);
    _v->addCoordinateSystem (1.0);
    // Create and add a dummy, empty point cloud so it can be updated later
    pcl::PointCloud<pcl::PointXYZ>::Ptr dummy_cloud (new pcl::PointCloud<pcl::PointXYZ>);
    _v->addPointCloud<pcl::PointXYZ>(dummy_cloud, "sample cloud");
    //_v->spinOnce(1);

}

getAndView::~getAndView()
{
}

void getAndView::lidarRawDataCallback(const sensor_msgs::PointCloud2::ConstPtr& msg)
{
   
    // Transform the message to a pcl::PCLPointCloud2
    pcl::PCLPointCloud2 pointCloud2;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_out (new pcl::PointCloud<pcl::PointXYZ>);
    pcl_conversions::toPCL(*msg, pointCloud2);

    ROS_INFO_STREAM("Data acquired");
    
    // Convert to plc cloud
    pcl::fromPCLPointCloud2(pointCloud2, *cloud_out);

    // Update the visualizer
    _v->updatePointCloud<pcl::PointXYZ>(cloud_out, "sample cloud"); // Has to have a same cloud ID as the cloud that needs to be updated, in this case the dummy one
    _v->spinOnce(1); // Refresh the visualizer
    //_v->spin();
    
}


int main(int argc, char **argv) {

    ros::init(argc, argv, "pcl_view_lidar");
    ros::Time::init();

    // Announce the start of the process
    ROS_INFO_STREAM("Data acquisition and visualization initiated!");

    // Initialize
    getAndView getAndViewBasic;

    ros::spin();

}