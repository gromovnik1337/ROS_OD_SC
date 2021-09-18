#!/usr/bin/env python3
import os
import sys
# Import dependencies
import open3d as o3d
import numpy as np
import rospy

def clean_pcd():
    # Load the point cloud data
    os.chdir(os.path.dirname(sys.argv[0]))
    point_cloud_in = o3d.io.read_point_cloud("../data/lidar_screenshot.pcd", print_progress = True)

    # Convert to numpy 
    point_cloud_in = np.asarray(point_cloud_in.points)
    point_cloud_clean = np.empty([0, 3])

    point_cloud_clean = np.asarray([p for p in point_cloud_in if (p[0]!=0 and p[1]!=0 and p[2]!=0)])

    point_cloud_out = o3d.geometry.PointCloud()
    point_cloud_out.points = o3d.utility.Vector3dVector(point_cloud_clean)
    o3d.io.write_point_cloud("../data/lidar_screenshot_clean.pcd", point_cloud_out, write_ascii = True)

if __name__ == '__main__':
    try:
        clean_pcd()
    except rospy.ROSInterruptException:
        pass