cmake_minimum_required(VERSION 2.6 FATAL_ERROR)

project(correspondence_grouping)

find_package(PCL 1.5 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable (correspondence_grouping_SHOT correspondence_grouping_SHOT.cpp)
target_link_libraries (correspondence_grouping_SHOT ${PCL_LIBRARIES})

#add_executable (correspondence_grouping_FPFH correspondence_grouping_FPFH.cpp)
#target_link_libraries (correspondence_grouping_FPFH ${PCL_LIBRARIES})

#add_executable (region_growing_segmentation region_growing_segmentation.cpp)
#target_link_libraries (region_growing_segmentation ${PCL_LIBRARIES})

#add_executable (conditional_euclidean_clustering conditional_euclidean_clustering.cpp)
#target_link_libraries (conditional_euclidean_clustering ${PCL_LIBRARIES})

#add_executable (region_growing_rgb_segmentation region_growing_rgb_segmentation.cpp)
#target_link_libraries (region_growing_rgb_segmentation ${PCL_LIBRARIES})

#add_executable (cylinder_segmentation cylinder_segmentation.cpp)
#target_link_libraries (cylinder_segmentation ${PCL_LIBRARIES})

#add_executable (cluster_extraction cluster_extraction.cpp)
#target_link_libraries (cluster_extraction ${PCL_LIBRARIES})

add_executable (pcd_size pcd_size.cpp)
target_link_libraries (pcd_size ${PCL_LIBRARIES})

#add_executable (directoryScan directoryScan.cpp)
#target_link_libraries (directoryScan ${PCL_LIBRARIES})

add_executable (correspondence_grouping_SHOT_Iterative correspondence_grouping_SHOT_Iterative.cpp)
target_link_libraries (correspondence_grouping_SHOT_Iterative ${PCL_LIBRARIES})

add_executable (objectExtractor objectExtractor.cpp)
target_link_libraries (objectExtractor ${PCL_LIBRARIES})

