# Required CMake version
cmake_minimum_required(VERSION 3.0.2)

# Name of the project
project(agitr)

# Find the catkin build system, and any other packaes on which the build depends
find_package(catkin REQUIRED COMPONENTS roscpp)
find_package(catkin REQUIRED COMPONENTS roscpp )

# Declare the catkin package
catkin_package()

# Specify the location of header files
# Specify the location of header files
include_directories(
    ${catkin_INCLUDE_DIRS} 
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src
)

# Declare the executable, along with its source files. For multiple executables multiple copies of this line 
# are to be used.
add_executable(hello ./src/hello.cpp)
add_executable(pubvel ./src/pubvel.cpp)
add_executable(subpose ./src/subpose.cpp)
add_executable(reverse_cmd_vel ./src/reverse_cmd_vel.cpp)
add_executable(set_bg_color ./src/set_bg_color.cpp)
add_executable(pubvel_with_max ./src/pubvel_with_max.cpp)
add_executable(spawn_turtle ./src/spawn_turtle.cpp)
add_executable(pubvel_toggle ./src/pubvel_toggle.cpp)

# Specify libraries aganist which to link. Again, this line is to be copied for each distinct executable in the package
target_link_libraries(hello ${catkin_LIBRARIES})
target_link_libraries(pubvel ${catkin_LIBRARIES})
target_link_libraries(subpose ${catkin_LIBRARIES})
target_link_libraries(reverse_cmd_vel ${catkin_LIBRARIES})
target_link_libraries(set_bg_color ${catkin_LIBRARIES})
target_link_libraries(pubvel_with_max ${catkin_LIBRARIES})
target_link_libraries(spawn_turtle ${catkin_LIBRARIES})
target_link_libraries(pubvel_toggle ${catkin_LIBRARIES})

