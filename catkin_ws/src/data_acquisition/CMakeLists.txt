# Required CMake version
cmake_minimum_required(VERSION 3.0.2)

# Name of the project
project(data_acquisition)

# Find the catkin build system, and any other packages on which the build depends
find_package(catkin REQUIRED COMPONENTS roscpp)
find_package(PCL 1.8 REQUIRED)

# Declare the catkin package
catkin_package(
    INCLUDE_DIRS
     ${PROJECT_SOURCE_DIR}/include
     ${PROJECT_SOURCE_DIR}/src
    DEPENDS
     PCL
)

catkin_package()

# Specify the location of header files
include_directories(
    ${catkin_INCLUDE_DIRS} 
    ${PCL_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src
)

link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# Declare the executable, along with its source files. For multiple executables multiple copies of this line 
# are to be used.
add_executable(acquire ./src/acquisition.cpp)
add_executable(pcl_lidar_screenshot ./src/pcl_lidar_screenshot.cpp)
add_executable(pcl_lidar_view ./src/pcl_lidar_view.cpp)
add_executable(range_image ./src/range_image.cpp)

# Specify libraries aganist which to link. Again, this line is to be copied for each distinct executable in the package
target_link_libraries(acquire ${catkin_LIBRARIES})
target_link_libraries(pcl_lidar_screenshot ${catkin_LIBRARIES} ${PCL_LIBRARIES})
target_link_libraries(pcl_lidar_view ${catkin_LIBRARIES} ${PCL_LIBRARIES})
target_link_libraries(range_image ${catkin_LIBRARIES} ${PCL_LIBRARIES})


