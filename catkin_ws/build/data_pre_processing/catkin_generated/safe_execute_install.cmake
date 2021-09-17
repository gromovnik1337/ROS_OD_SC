execute_process(COMMAND "/home/luxc/vice_ROS_OD_SC/catkin_ws/build/data_pre_processing/catkin_generated/python_distutils_install.sh" RESULT_VARIABLE res)

if(NOT res EQUAL 0)
  message(FATAL_ERROR "execute_process(/home/luxc/vice_ROS_OD_SC/catkin_ws/build/data_pre_processing/catkin_generated/python_distutils_install.sh) returned error code ")
endif()
