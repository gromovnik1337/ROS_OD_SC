// This script acquires the Livox Mid 40 data and subsequently outputs the .pcd file of a single frame and 
// visualizes a single frame using PCL viewer
// Created by: Vice, 09.04.2021

#include <pcl_lidar_screenshot.hpp>


getAndView::getAndView()
{

    _cloud_subscriber = _node_handle.subscribe<sensor_msgs::PointCloud2>("/livox/lidar", 1, &getAndView::lidarRawDataCallback, this);  

}

getAndView::~getAndView()
{
}

pcl::visualization::PCLVisualizer::Ptr simpleVis (pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud)
{

    pcl::visualization::PCLVisualizer::Ptr _v (new pcl::visualization::PCLVisualizer("Viewer"));
    _v->setBackgroundColor (0, 0, 0);
    _v->addPointCloud<pcl::PointXYZ>(cloud, "sample cloud");
    _v->addCoordinateSystem (1.0);
    _v->spin();
    return (_v);
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
    // Write out the cloud
    // It is important to write out the cloud before it is being sent to visualizer
    pcl::io::savePCDFileASCII("lidar_screenshot.pcd", *cloud_out);
    ROS_INFO_STREAM("Cloud saved");

    pcl::visualization::PCLVisualizer::Ptr viewer;
    viewer = simpleVis(cloud_out);

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