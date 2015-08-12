cmake_minimum_required(VERSION 2.6 FATAL_ERROR)

project(correspondence_grouping)

find_package(PCL 1.5 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable (correspondence_grouping_SHOT  src/correspondence_grouping_SHOT.cpp)
target_link_libraries (correspondence_grouping_SHOT ${PCL_LIBRARIES})

add_executable (correspondence_grouping_SHOT_Obj-Obj src/correspondence_grouping_SHOT_Obj-Obj.cpp)
target_link_libraries (correspondence_grouping_SHOT_Obj-Obj ${PCL_LIBRARIES})

#add_executable (correspondence_grouping_FPFH  src/correspondence_grouping_FPFH.cpp)
#target_link_libraries (correspondence_grouping_FPFH ${PCL_LIBRARIES})

#add_executable (region_growing_segmentation  src/region_growing_segmentation.cpp)
#target_link_libraries (region_growing_segmentation ${PCL_LIBRARIES})

#add_executable (conditional_euclidean_clustering  src/conditional_euclidean_clustering.cpp)
#target_link_libraries (conditional_euclidean_clustering ${PCL_LIBRARIES})

#add_executable (region_growing_rgb_segmentation  src/region_growing_rgb_segmentation.cpp)
#target_link_libraries (region_growing_rgb_segmentation ${PCL_LIBRARIES})

#add_executable (cylinder_segmentation  src/cylinder_segmentation.cpp)
#target_link_libraries (cylinder_segmentation ${PCL_LIBRARIES})

#add_executable (cluster_extraction  src/cluster_extraction.cpp)
#target_link_libraries (cluster_extraction ${PCL_LIBRARIES})

#add_executable (cluster_extraction_v2  src/cluster_extraction_v2.cpp)
#target_link_libraries (cluster_extraction_v2 ${PCL_LIBRARIES})

#add_executable (directoryScan  src/directoryScan.cpp)
#target_link_libraries (directoryScan ${PCL_LIBRARIES})

#add_executable (correspondence_grouping_SHOT_Iterative  src/correspondence_grouping_SHOT_Iterative.cpp)
#target_link_libraries (correspondence_grouping_SHOT_Iterative ${PCL_LIBRARIES})

add_executable (pcd_size  src/pcd_size.cpp)
target_link_libraries (pcd_size ${PCL_LIBRARIES})

add_executable (objectExtractor  src/objectExtractor.cpp)
target_link_libraries (objectExtractor ${PCL_LIBRARIES})

add_executable (correspondence_grouping_SHOT_Iterative_v2 src/correspondence_grouping_SHOT_Iterative_v2.cpp)
target_link_libraries (correspondence_grouping_SHOT_Iterative_v2 ${PCL_LIBRARIES})

add_executable (correspondence_grouping_CSHOT src/correspondence_grouping_CSHOT.cpp)
target_link_libraries (correspondence_grouping_CSHOT ${PCL_LIBRARIES})

#add_executable (3DSIFT_Keypoints src/3DSIFT_Keypoints.cpp)
#target_link_libraries (3DSIFT_Keypoints ${PCL_LIBRARIES})

add_executable (correspondence_grouping_SHOT_Iterative_v3 src/correspondence_grouping_SHOT_Iterative_v3.cpp)
target_link_libraries (correspondence_grouping_SHOT_Iterative_v3 ${PCL_LIBRARIES})

#add_executable (transform_demo src/transform_demo.cpp)
#target_link_libraries (transform_demo ${PCL_LIBRARIES})

add_executable (objectExtractorIterative src/objectExtractorIterative.cpp)
target_link_libraries (objectExtractorIterative ${PCL_LIBRARIES})

add_executable (correspondence_grouping_SHOT_Iterative_Obj-Obj  src/correspondence_grouping_SHOT_Iterative_Obj-Obj.cpp)
target_link_libraries (correspondence_grouping_SHOT_Iterative_Obj-Obj ${PCL_LIBRARIES})





