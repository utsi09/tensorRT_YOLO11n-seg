cmake_minimum_required(VERSION 3.8)
project(lidar_yolo_fusion)

# ==============================================================================
# [TensorRT 라이브러리 경로 강제 할당]
# - 링커가 찾지 못하는 경로 대신 직접 라이브러리 파일을 지정
# ==============================================================================
set(TENSORRT_LIBRARY "/usr/lib/x86_64-linux-gnu/libnvinfer.so")
set(TENSORRT_PLUGIN_LIBRARY "/usr/lib/x86_64-linux-gnu/libnvinfer_plugin.so")
# ==============================================================================

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# C++ 17 표준 사용
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 1. 의존성 패키지 찾기
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(message_filters REQUIRED)
find_package(pcl_conversions REQUIRED)

# 2. 시스템 라이브러리 찾기 (PCL, OpenCV, CUDA)
find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)
find_package(Eigen3 REQUIRED)

# PCL 관련 정의 포함
add_definitions(${PCL_DEFINITIONS})
link_directories(${PCL_LIBRARY_DIRS})

# 3. 실행 파일 생성
add_executable(lidar_yolo_fusion_node src/lidar_yolo_fusion_node.cpp)

# 4. Include 경로 설정 (헤더 파일 경로)
target_include_directories(lidar_yolo_fusion_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  
  # ROS 2 INCLUDE_DIRS
  ${rclcpp_INCLUDE_DIRS}
  ${cv_bridge_INCLUDE_DIRS}
  ${message_filters_INCLUDE_DIRS}
  ${pcl_conversions_INCLUDE_DIRS}
  
  # 시스템/외부 라이브러리
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}          
  /usr/local/cuda/include     
  /usr/include/x86_64-linux-gnu  
)

# 5. [★ 필수] ROS 2 의존성을 ament에 위임 (Linking)
ament_target_dependencies(lidar_yolo_fusion_node
  rclcpp
  sensor_msgs
  geometry_msgs
  tf2_ros
  cv_bridge
  message_filters
  pcl_conversions
)

# 6. [★ 시스템 라이브러리만 링킹 ★]
target_link_libraries(lidar_yolo_fusion_node
  # 1. PCL, OpenCV (시스템 모듈)
  ${PCL_LIBRARIES}
  ${OpenCV_LIBRARIES}
  opencv_core
  opencv_imgproc
  opencv_dnn 

  # 2. TENSORRT & CUDA LIBRARIES
  ${CUDA_LIBRARIES}
  ${TENSORRT_LIBRARY}          
  ${TENSORRT_PLUGIN_LIBRARY}   
)

install(TARGETS lidar_yolo_fusion_node
  DESTINATION lib/${PROJECT_NAME})

ament_package()