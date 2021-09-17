# Required CMake version
cmake_minimum_required(VERSION 3.0.2)

# Name of the project
project(data_pre_processing)

find_package(catkin REQUIRED COMPONENTS rospy)

catkin_package()
# N.B. The script that catkin_install_python is invoked upon must be made executable with: "chmod 755 node.py"
catkin_install_python(PROGRAMS 
src/simple_publisher.py 
src/simple_subscriber.py
src/clean_point_cloud.py    

DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
