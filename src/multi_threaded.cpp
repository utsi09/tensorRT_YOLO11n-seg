#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <map>
#include <mutex>
#include <memory>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "tf2_ros/static_transform_broadcaster.h"
#include "cv_bridge/cv_bridge.h"
#include "pcl_conversions/pcl_conversions.h"
#include "pcl/point_cloud.h"
#include "pcl/point_types.h"
#include "message_filters/subscriber.h"
#include "message_filters/synchronizer.h"
#include "message_filters/sync_policies/approximate_time.h"

#include "lidar_yolo_fusion/msg/obstacle_info.hpp"
#include "lidar_yolo_fusion/msg/obstacle_array.hpp"

#include <Eigen/Dense>

#include <cuda_runtime_api.h>
#include <NvInfer.h>
#include <opencv2/opencv.hpp>

#include <future>
#include <thread>

using namespace std;
using namespace cv;
using namespace std::chrono; 

const int INPUT_W = 640;
const int INPUT_H = 640;
const int NUM_CLASSES = 80;
const int NUM_MASKS = 32;
const int OUTPUT_CHANNELS = 4 + NUM_CLASSES + NUM_MASKS;
const int OUTPUT_GRID = 8400;

struct Detection {
    int class_id;
    float conf;
    cv::Rect box;
    cv::Mat mask;
};

struct CameraProcessResult {
    int camera_index;
    cv::Mat overlay_image;
    std::vector<Eigen::Vector3f> centers;
    std::vector<Eigen::Vector3f> closest_points;
    std::vector<std::string> labels;
    std::vector<int> class_ids;
    std::vector<float> dists;
    std::vector<int> object_point_indices;  // 객체에 속하는 포인트 인덱스
};

class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) cout << "[TRT] " << msg << std::endl;
    }
} gLogger;

class YoloDetector {
public:
    YoloDetector(const std::string& engine_path, float conf_thres, float iou_thres)
        : conf_thres_(conf_thres), iou_thres_(iou_thres), infer_mutex_(std::make_unique<std::mutex>())
    {
        std::ifstream file(engine_path, std::ios::binary);
        if (!file.good()) {
            std::cerr << "Error: Engine file not found: " << engine_path << std::endl;
            return;
        }
        file.seekg(0, file.end);
        size_t size = file.tellg();
        file.seekg(0, file.beg);
        std::vector<char> engine_data(size);
        file.read(engine_data.data(), size);
        file.close();

        runtime_ = nvinfer1::createInferRuntime(gLogger);
        engine_ = runtime_->deserializeCudaEngine(engine_data.data(), size);
        context_ = engine_->createExecutionContext();
        
        // 스트림 생성 (성능 위해)
        cudaStreamCreate(&stream_);

        // 버퍼 할당
        cudaMalloc(&buffers_[0], INPUT_W * INPUT_H * 3 * sizeof(float));         // Input
        cudaMalloc(&buffers_[1], OUTPUT_CHANNELS * OUTPUT_GRID * sizeof(float)); // Output0 (Box+Class+MaskCoeff)
        cudaMalloc(&buffers_[2], NUM_MASKS * 160 * 160 * sizeof(float));         // Output1 (Proto Masks)

        std::cout << "TensorRT Engine Loaded with Segmentation Support!" << std::endl;
    }

    ~YoloDetector() {
        cudaFree(buffers_[0]);
        cudaFree(buffers_[1]);
        cudaFree(buffers_[2]);
        cudaStreamDestroy(stream_); // 스트림 해제
        delete context_;
        delete engine_;
        delete runtime_;
    }

    std::vector<Detection> detect(const cv::Mat& img) {
        std::vector<Detection> results;
        if (!context_ || img.empty()) return results;

        // CUDA/TensorRT 추론을 위한 mutex lock (스레드 안전성)
        std::lock_guard<std::mutex> lock(*infer_mutex_);

        // 1. 전처리
        cv::Mat blob;
        cv::dnn::blobFromImage(img, blob, 1.0 / 255.0, cv::Size(INPUT_W, INPUT_H), cv::Scalar(), true, false);
        cudaMemcpyAsync(buffers_[0], blob.ptr<float>(), blob.total() * sizeof(float), cudaMemcpyHostToDevice, stream_);

        // 2. 바인딩 및 실행
        context_->setInputTensorAddress("images", buffers_[0]);
        context_->setOutputTensorAddress("output0", buffers_[1]);
        context_->setOutputTensorAddress("output1", buffers_[2]);
        context_->enqueueV3(stream_);

        // 3. 결과 복사
        std::vector<float> output0_data(OUTPUT_CHANNELS * OUTPUT_GRID);
        std::vector<float> output1_data(NUM_MASKS * 160 * 160); // Proto masks

        cudaMemcpyAsync(output0_data.data(), buffers_[1], output0_data.size() * sizeof(float), cudaMemcpyDeviceToHost, stream_);
        cudaMemcpyAsync(output1_data.data(), buffers_[2], output1_data.size() * sizeof(float), cudaMemcpyDeviceToHost, stream_);
        cudaStreamSynchronize(stream_); // 동기화

        // 4. 후처리 준비
        float x_scale = (float)img.cols / INPUT_W;
        float y_scale = (float)img.rows / INPUT_H;

        std::vector<int> class_ids;
        std::vector<float> confidences;
        std::vector<cv::Rect> boxes;
        std::vector<std::vector<float>> mask_coeffs; // 마스크 계수 저장용

        cv::Mat output0_mat(OUTPUT_CHANNELS, OUTPUT_GRID, CV_32F, output0_data.data());

        for (int i = 0; i < OUTPUT_GRID; ++i) {
            float max_conf = 0.0;
            int max_class_id = -1;

            // 클래스 스코어 확인
            for (int c = 0; c < NUM_CLASSES; ++c) {
                float score = output0_mat.at<float>(4 + c, i);
                if (score > max_conf) {
                    max_conf = score;
                    max_class_id = c;
                }
            }

            if (max_conf > conf_thres_) {
                // Box Parsing
                float cx = output0_mat.at<float>(0, i);
                float cy = output0_mat.at<float>(1, i);
                float w = output0_mat.at<float>(2, i);
                float h = output0_mat.at<float>(3, i);

                int left = int((cx - 0.5 * w) * x_scale);
                int top = int((cy - 0.5 * h) * y_scale);
                int width = int(w * x_scale);
                int height = int(h * y_scale);
                
                // 마스크 계수 추출 (마지막 32개 채널)
                std::vector<float> coeffs(NUM_MASKS);
                for (int m = 0; m < NUM_MASKS; ++m) {
                    coeffs[m] = output0_mat.at<float>(4 + NUM_CLASSES + m, i);
                }

                boxes.push_back(cv::Rect(left, top, width, height));
                class_ids.push_back(max_class_id);
                confidences.push_back(max_conf);
                mask_coeffs.push_back(coeffs);
            }
        }

        // NMS
        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxes, confidences, conf_thres_, iou_thres_, indices);

        // 5. 마스크 디코딩 (행렬 연산)
        // Proto Mask: (32, 160, 160)
        cv::Mat proto(NUM_MASKS, 160 * 160, CV_32F, output1_data.data());

        for (int idx : indices) {
            Detection det;
            det.class_id = class_ids[idx];
            det.conf = confidences[idx];
            det.box = boxes[idx] & cv::Rect(0, 0, img.cols, img.rows); // 이미지 범위 클리핑

            if (det.box.area() > 0) {
                // (1) 마스크 생성: Coeffs(1x32) * Proto(32x25600) = (1x25600)
                cv::Mat coeff_mat(1, NUM_MASKS, CV_32F, mask_coeffs[idx].data());
                cv::Mat mask_logits = coeff_mat * proto; // 행렬 곱
                
                // (2) Reshape to 160x160
                cv::Mat mask_proto = mask_logits.reshape(1, 160); 

                // (3) Sigmoid
                cv::exp(-mask_proto, mask_proto);
                mask_proto = 1.0 / (1.0 + mask_proto);

                // (4) Resize to Original Image Size
                cv::Mat mask_resized;
                cv::resize(mask_proto, mask_resized, cv::Size(INPUT_W, INPUT_H)); // 640x640

                // (5) Crop to Bounding Box (Scale 고려)
                // 마스크 좌표계는 640x640 기준이므로, 원본 이미지 좌표계로 변환 필요
                // 편의상 640x640으로 리사이즈된 마스크에서 원본 비율에 맞는 ROI를 따야 함.
                // 여기서는 간단히 전체 이미지를 원본 크기로 리사이즈 후 자릅니다.
                cv::resize(mask_resized, mask_resized, img.size());
                
                // (6) Thresholding (> 0.5) & Binarize
                // 메모리 효율을 위해 Box 영역만 잘라내서 저장
                cv::Mat final_mask = mask_resized(det.box) > 0.5;
                
                // CV_8UC1 형태로 변환 (0 or 255)
                final_mask.convertTo(det.mask, CV_8UC1, 255);
                
                results.push_back(det);
            }
        }
        return results;
    }

private:
    float conf_thres_;
    float iou_thres_;
    cudaStream_t stream_;
    nvinfer1::IRuntime* runtime_ = nullptr;
    nvinfer1::ICudaEngine* engine_ = nullptr;
    nvinfer1::IExecutionContext* context_ = nullptr;
    void* buffers_[3];
    std::unique_ptr<std::mutex> infer_mutex_;  // 스레드 안전성을 위한 mutex
};

class LidarYoloFusionNode : public rclcpp::Node {
public:
    LidarYoloFusionNode() : Node("lidar_yolo_fusion") {
        declare_parameter("lidar_topic", "/carla/hero/lidar");
        declare_parameter("model_path", "yolo11n-seg.engine");
        declare_parameter("conf", 0.3);
        declare_parameter("iou", 0.5);

        std::string lidar_topic = get_parameter("lidar_topic").as_string();
        std::string model_path = get_parameter("model_path").as_string();
        float conf = get_parameter("conf").as_double();
        float iou = get_parameter("iou").as_double();

        detectors_.resize(4);
        for (int i=0; i<4; i++){
            detectors_[i] = std::make_shared<YoloDetector>(model_path, conf, iou);
        }

        //detector_ = std::make_shared<YoloDetector>(model_path, conf, iou);

        tf_broadcaster_ = std::make_shared<tf2_ros::StaticTransformBroadcaster>(this);
        publish_static_tf();

        setup_transforms();

        pub_overlay_front_ = this->create_publisher<sensor_msgs::msg::Image>("lidar_yolo/overlay/front", 10);
        pub_overlay_left_ = this->create_publisher<sensor_msgs::msg::Image>("lidar_yolo/overlay/left", 10);
        pub_overlay_right_ = this->create_publisher<sensor_msgs::msg::Image>("lidar_yolo/overlay/right", 10);
        pub_overlay_back_ = this->create_publisher<sensor_msgs::msg::Image>("lidar_yolo/overlay/back", 10);

        overlay_pubs_.push_back(pub_overlay_front_);
        overlay_pubs_.push_back(pub_overlay_left_);
        overlay_pubs_.push_back(pub_overlay_right_);
        overlay_pubs_.push_back(pub_overlay_back_);

        pub_obj_pc_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("lidar_yolo/object_pointcloud", 10);
        pub_topview_ = this->create_publisher<sensor_msgs::msg::Image>("lidar_yolo/topview", 10);
        pub_obstacles_ = this->create_publisher<lidar_yolo_fusion::msg::ObstacleArray>("lidar_yolo/obstacles", 10);

        // Odometry subscriber
        sub_odom_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "/carla/hero/odometry", 10,
            std::bind(&LidarYoloFusionNode::odom_callback, this, std::placeholders::_1));

        sub_lidar_.subscribe(this, lidar_topic);
        sub_front_.subscribe(this, "/carla/hero/rgb_front/image");
        sub_left_.subscribe(this, "/carla/hero/rgb_left/image");
        sub_right_.subscribe(this, "/carla/hero/rgb_right/image");
        sub_back_.subscribe(this, "/carla/hero/rgb_back/image");

        sync_ = std::make_shared<Sync>(SyncPolicy(10), sub_lidar_, sub_front_, sub_left_, sub_right_, sub_back_);
        sync_->registerCallback(std::bind(&LidarYoloFusionNode::sync_callback, this,
            std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4, std::placeholders::_5));

        class_names_[0] = "person"; class_names_[2] = "car"; class_names_[5] = "bus"; class_names_[7] = "truck";
    }

private:
    CameraProcessResult process_single_camera(
        int index,
        const sensor_msgs::msg::Image::ConstSharedPtr& img_msg,
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
        const Eigen::Matrix4f& extrinsic,
        const Eigen::Matrix3f& K
    )
    {
        CameraProcessResult result;
        result.camera_index = index;
        
        // 1. 이미지 변환
        cv::Mat cv_img;
        try {
            cv_img = cv_bridge::toCvCopy(img_msg, "bgr8")->image.clone();
        } catch (cv_bridge::Exception& e){
            return result; 
        }

        // 2. YOLO 추론 (해당 인덱스의 detector 사용)
        std::vector<Detection> detections = detectors_[index]->detect(cv_img);

        // 3. 라이다 투영 (CPU 부하가 큰 부분)
        std::vector<cv::Point> valid_uvs;
        std::vector<int> valid_indices;

        for (size_t k = 0; k < cloud->points.size(); k++) {
            const auto& pt = cloud->points[k];
            Eigen::Vector4f p_lidar(pt.x, pt.y, pt.z, 1.0f);
            Eigen::Vector4f p_cam = extrinsic * p_lidar;

            if (p_cam(2) <= 0) continue; 
            Eigen::Vector3f proj = K * p_cam.head<3>();
            if (proj(2) == 0) continue;

            int u = static_cast<int>(proj(0) / proj(2));
            int v = static_cast<int>(proj(1) / proj(2));

            if (u >= 0 && u < cv_img.cols && v >= 0 && v < cv_img.rows) {
                valid_uvs.emplace_back(u, v);
                valid_indices.push_back(k);
            }
        }

        // 4. 객체 매칭 및 마스크 처리
        std::vector<int> target_classes = {0, 2, 5, 7}; // person, car, bus, truck

        for (auto& det : detections) {
            if (det.box.area() <= 0) continue;

            // 시각화 (Overlay 그리기)
            cv::rectangle(cv_img, det.box, cv::Scalar(0, 255, 0), 2);
            std::string label = class_names_[det.class_id];
            cv::putText(cv_img, label, det.box.tl(), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);

            // 마스크 오버레이
            if (!det.mask.empty()) {
                            cv::Mat colored_mask;
                            cv::cvtColor(det.mask, colored_mask, cv::COLOR_GRAY2BGR);
                            colored_mask.setTo(cv::Scalar(0, 0, 255), det.mask > 0); 
                            cv::addWeighted(cv_img(det.box), 1.0, colored_mask(cv::Rect(0,0,det.box.width, det.box.height)), 0.4, 0.0, cv_img(det.box));
                        }

            bool is_target = false;
            
            for(int t_id : target_classes) {
                if(det.class_id == t_id) { is_target = true; break; }
            }

            if (is_target && !det.mask.empty()) {
                 // 마스크 축소 로직 (0.8배)
                cv::Mat final_check_mask;
                float scale_ratio = 0.8f;
                int new_w = static_cast<int>(det.mask.cols * scale_ratio);
                int new_h = static_cast<int>(det.mask.rows * scale_ratio);

                if (new_w > 0 && new_h > 0) {
                    final_check_mask = cv::Mat::zeros(det.mask.size(), det.mask.type());
                    cv::Mat resized_content;
                    cv::resize(det.mask, resized_content, cv::Size(new_w, new_h), 0, 0, cv::INTER_NEAREST);
                    int offset_x = (det.mask.cols - new_w) / 2;
                    int offset_y = (det.mask.rows - new_h) / 2;
                    resized_content.copyTo(final_check_mask(cv::Rect(offset_x, offset_y, new_w, new_h)));
                } else {
                    final_check_mask = det.mask; 
                }

                // 포인트 매칭
                Eigen::Vector3f sum_points(0, 0, 0);
                Eigen::Vector3f closest_point(0, 0, 0);
                int count_points = 0;
                float min_dist = 9999.0f;

                for (size_t k = 0; k < valid_uvs.size(); ++k) {
                    cv::Point pt = valid_uvs[k];
                    if (det.box.contains(pt)) {
                        int mx = pt.x - det.box.x;
                        int my = pt.y - det.box.y;

                        if (mx >= 0 && mx < final_check_mask.cols && my >= 0 && my < final_check_mask.rows) {
                            if (final_check_mask.at<uint8_t>(my, mx) > 0) {
                                int original_idx = valid_indices[k];
                                const auto& p3d = cloud->points[original_idx];
                                Eigen::Vector3f p_vec(p3d.x, p3d.y, p3d.z);

                                float dist = p_vec.norm();
                                if (dist < min_dist) {
                                    min_dist = dist;
                                    closest_point = p_vec;
                                }
                                sum_points += p_vec;
                                count_points++;

                                // 객체 포인트 인덱스 저장
                                result.object_point_indices.push_back(original_idx);
                            }
                        }
                    }
                }

                if (count_points > 0) {
                    Eigen::Vector3f center = sum_points / static_cast<float>(count_points);
                    result.centers.push_back(center);
                    result.closest_points.push_back(closest_point);
                    result.labels.push_back(label);
                    result.class_ids.push_back(det.class_id);
                    result.dists.push_back(min_dist);
                }
            }
        }
        
        result.overlay_image = cv_img;
        return result;
    }
    // Odometry callback - store latest ego pose
    void odom_callback(const nav_msgs::msg::Odometry::ConstSharedPtr& msg) {
        std::lock_guard<std::mutex> lock(odom_mutex_);
        ego_x_ = msg->pose.pose.position.x;
        ego_y_ = msg->pose.pose.position.y;
        ego_z_ = msg->pose.pose.position.z;

        // Extract yaw from quaternion
        double qx = msg->pose.pose.orientation.x;
        double qy = msg->pose.pose.orientation.y;
        double qz = msg->pose.pose.orientation.z;
        double qw = msg->pose.pose.orientation.w;

        // yaw (z-axis rotation)
        double siny_cosp = 2.0 * (qw * qz + qx * qy);
        double cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz);
        ego_yaw_ = std::atan2(siny_cosp, cosy_cosp);

        odom_received_ = true;
    }

    // Transform point from lidar frame to Carla world frame
    Eigen::Vector3d lidar_to_world(const Eigen::Vector3f& p_lidar) {
        double x_ego, y_ego, yaw;
        {
            std::lock_guard<std::mutex> lock(odom_mutex_);
            x_ego = ego_x_;
            y_ego = ego_y_;
            yaw = ego_yaw_;
        }

        // Lidar is mounted at offset from vehicle center
        // Lidar frame: X=forward, Y=left, Z=up (in vehicle frame)
        // Carla world: X=forward, Y=left, Z=up

        double cos_yaw = std::cos(yaw);
        double sin_yaw = std::sin(yaw);

        // Rotate lidar point to world frame and add ego position
        double world_x = x_ego + cos_yaw * p_lidar(0) - sin_yaw * p_lidar(1);
        double world_y = y_ego + sin_yaw * p_lidar(0) + cos_yaw * p_lidar(1);
        double world_z = p_lidar(2);  // Z stays same (assuming flat ground)

        return Eigen::Vector3d(world_x, world_y, world_z);
    }

    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_overlay_front_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_overlay_left_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_overlay_right_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_overlay_back_;

    void setup_transforms() {
        Eigen::Matrix4f T_front;
        T_front << 0, -1, 0, 0.0,
            0, 0, -1, -0.4,
            1, 0, 0, -2.0,
            0, 0, 0, 1.0;

        Eigen::Matrix4f T_left;
        T_left << 1, 0, 0, 0.0,
            0, 0, -1, -1.4,
            0, 1, 0, -1.0,
            0, 0, 0, 1.0;

        Eigen::Matrix4f T_right;
        T_right << -1, 0, 0, 0.0,
            0, 0, -1, -1.4,
            0, -1, 0, -1.0,
            0, 0, 0, 1.0;

        Eigen::Matrix4f T_back;
        T_back << 0, 1, 0, 0.0,
            0, 0, -1, -0.4,
            -1, 0, 0, -2.0,
            0, 0, 0, 1.0;

        extrinsic_list_ = {T_front, T_left, T_right, T_back};

        for (auto& T : extrinsic_list_) {
            cam2lidar_list_.push_back(T.inverse());
        }

        K_ << 512.0, 0.0, 512.0,
            0.0, 512.0, 384.0,
            0.0, 0.0, 1.0;
    }

    void publish_static_tf() {
        geometry_msgs::msg::TransformStamped t;
        t.header.stamp = this->now();
        t.header.frame_id = "rgb_front";
        t.child_frame_id = "lidar";
        t.transform.translation.x = -2.0;
        t.transform.translation.y = 0.0;
        t.transform.translation.z = 0.4;
        t.transform.rotation.w = 1.0;
        tf_broadcaster_->sendTransform(t);
    }

    void sync_callback(
                const sensor_msgs::msg::PointCloud2::ConstSharedPtr& lidar_msg,
                const sensor_msgs::msg::Image::ConstSharedPtr& front_msg,
                const sensor_msgs::msg::Image::ConstSharedPtr& left_msg,
                const sensor_msgs::msg::Image::ConstSharedPtr& right_msg,
                const sensor_msgs::msg::Image::ConstSharedPtr& back_msg)
        {
            auto start_total = high_resolution_clock::now();
            
            // 1. 라이다 데이터 변환 (공통 작업)
            pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
            pcl::fromROSMsg(*lidar_msg, *cloud);
            if (cloud->empty()) return;

            std::vector<sensor_msgs::msg::Image::ConstSharedPtr> img_msgs = {front_msg, left_msg, right_msg, back_msg};
            
            // 2. 비동기 작업 시작 (4개 카메라 병렬 실행)
            std::vector<std::future<CameraProcessResult>> futures;
            for (int i = 0; i < 4; i++) {
                futures.push_back(std::async(std::launch::async, 
                    &LidarYoloFusionNode::process_single_camera, 
                    this, 
                    i, 
                    img_msgs[i], 
                    cloud, 
                    extrinsic_list_[i], 
                    K_
                ));
            }

            // 3. 결과 수집 및 병합
            std::vector<Eigen::Vector3f> all_centers;
            std::vector<Eigen::Vector3f> all_closest_points;
            std::vector<std::string> all_labels;
            std::vector<int> all_class_ids;
            std::vector<float> all_dists;
            
            std::vector<bool> is_object_point(cloud->points.size(), false); // (옵션: 포인트 클라우드 시각화용, 필요 시 로직 추가 필요)

            for (int i = 0; i < 4; i++) {
                // 스레드가 끝날 때까지 대기하고 결과 받음
                CameraProcessResult res = futures[i].get();

                // Overlay Publish
                if (!res.overlay_image.empty()) {
                    sensor_msgs::msg::Image::SharedPtr out_msg = 
                        cv_bridge::CvImage(img_msgs[i]->header, "bgr8", res.overlay_image).toImageMsg();
                    overlay_pubs_[i]->publish(*out_msg);
                }

                // 데이터 병합
                all_centers.insert(all_centers.end(), res.centers.begin(), res.centers.end());
                all_closest_points.insert(all_closest_points.end(), res.closest_points.begin(), res.closest_points.end());
                all_labels.insert(all_labels.end(), res.labels.begin(), res.labels.end());
                all_class_ids.insert(all_class_ids.end(), res.class_ids.begin(), res.class_ids.end());
                all_dists.insert(all_dists.end(), res.dists.begin(), res.dists.end());

                // 객체 포인트 인덱스 표시
                for (int idx : res.object_point_indices) {
                    if (idx >= 0 && idx < static_cast<int>(is_object_point.size())) {
                        is_object_point[idx] = true;
                    }
                }
            }

            // 객체 포인트클라우드 발행
            pcl::PointCloud<pcl::PointXYZ> obj_cloud_accum;
            for (size_t k = 0; k < cloud->points.size(); k++) {
                if (is_object_point[k]) {
                    obj_cloud_accum.push_back(cloud->points[k]);
                }
            }

            if (!obj_cloud_accum.empty()) {
                sensor_msgs::msg::PointCloud2 pc_msg;
                pcl::toROSMsg(obj_cloud_accum, pc_msg);
                pc_msg.header = lidar_msg->header;
                pc_msg.header.frame_id = "hero/lidar";
                pub_obj_pc_->publish(pc_msg);
            }

            // 4. TopView 및 Obstacle Array 발행 (기존 로직 유지)
            if (!all_centers.empty()) {
                cv::Mat top_view = build_topview(all_centers, all_labels, all_dists);
                sensor_msgs::msg::Image::SharedPtr top_msg = cv_bridge::CvImage(front_msg->header, "bgr8", top_view).toImageMsg();
                pub_topview_->publish(*top_msg);
            }

            if (odom_received_ && !all_closest_points.empty()) {
                lidar_yolo_fusion::msg::ObstacleArray obs_array;
                obs_array.header = lidar_msg->header;
                obs_array.header.frame_id = "map";

                {
                    std::lock_guard<std::mutex> lock(odom_mutex_);
                    obs_array.ego_x = ego_x_;
                    obs_array.ego_y = ego_y_;
                    obs_array.ego_yaw = ego_yaw_;
                }

                for (size_t i = 0; i < all_closest_points.size(); i++) {
                    lidar_yolo_fusion::msg::ObstacleInfo obs;
                    obs.label = all_labels[i];
                    obs.class_id = all_class_ids[i];
                    obs.rel_x = all_closest_points[i](0);
                    obs.rel_y = all_closest_points[i](1);
                    obs.distance = all_dists[i];

                    Eigen::Vector3d world_pos = lidar_to_world(all_closest_points[i]);
                    obs.x = world_pos(0);
                    obs.y = world_pos(1);
                    obs.z = world_pos(2);
                    obs.velocity_x = 0.0;
                    obs.velocity_y = 0.0;

                    obs_array.obstacles.push_back(obs);
                }
                pub_obstacles_->publish(obs_array);
            }

            auto end_total = high_resolution_clock::now();
            auto duration_total = duration_cast<milliseconds>(end_total - start_total);
            RCLCPP_INFO(this->get_logger(), "[CPP_PERF] Total Parallel Execution: %ld ms", duration_total.count());
        }

    cv::Mat build_topview(
        const std::vector<Eigen::Vector3f>& centers,
        const std::vector<std::string>& labels,
        const std::vector<float>& dists)
    {
        // std::cout << "[DEBUG] build_topview started. Centers count: " << centers.size() << std::endl; // 디버그 로그 주석 처리

        int width = 600;
        int height = 600;
        cv::Mat top_view = cv::Mat::zeros(height, width, CV_8UC3);

        float max_range = 20.0f;
        int cx = width / 2;
        int cy = height / 2;
        float scale = (float)width / (2.0f * max_range);

        cv::line(top_view, cv::Point(cx, 0), cv::Point(cx, height), cv::Scalar(100, 100, 100), 1);
        cv::line(top_view, cv::Point(0, cy), cv::Point(width, cy), cv::Scalar(100, 100, 100), 1);
        for (int r = 5; r <= max_range; r += 5) {
            cv::circle(top_view, cv::Point(cx, cy), int(r * scale), cv::Scalar(50, 50, 50), 1);
        }

        for (size_t i = 0; i < centers.size(); i++) {
            if (i >= labels.size() || i >= dists.size()) {
                continue;
            }

            const auto& c = centers[i];
            int px = cx - int(c(1) * scale);
            int py = cy - int(c(0) * scale);

            if (px >= 0 && px < width && py >= 0 && py < height) {
                cv::circle(top_view, cv::Point(px, py), 5, cv::Scalar(0, 0, 255), -1);

                std::string text = labels[i];
                if (dists[i] > 0.0f) {
                    std::stringstream ss;
                    ss << std::fixed << std::setprecision(1) << dists[i];
                    text += " " + ss.str() + "m";
                }
                cv::putText(top_view, text, cv::Point(px + 7, py),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
            }
        }
        return top_view;
    }

    using SyncPolicy = message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::PointCloud2, sensor_msgs::msg::Image, sensor_msgs::msg::Image, sensor_msgs::msg::Image, sensor_msgs::msg::Image>;
    using Sync = message_filters::Synchronizer<SyncPolicy>;

    std::vector<std::shared_ptr<YoloDetector>> detectors_;
    // std::shared_ptr<YoloDetector> detector_;
    std::shared_ptr<tf2_ros::StaticTransformBroadcaster> tf_broadcaster_;
    message_filters::Subscriber<sensor_msgs::msg::PointCloud2> sub_lidar_;
    message_filters::Subscriber<sensor_msgs::msg::Image> sub_front_, sub_left_, sub_right_, sub_back_;
    std::shared_ptr<Sync> sync_;

    std::vector<rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr> overlay_pubs_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_obj_pc_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_topview_;
    rclcpp::Publisher<lidar_yolo_fusion::msg::ObstacleArray>::SharedPtr pub_obstacles_;

    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr sub_odom_;

    std::vector<Eigen::Matrix4f> extrinsic_list_;
    std::vector<Eigen::Matrix4f> cam2lidar_list_;
    Eigen::Matrix3f K_;
    std::map<int, std::string> class_names_;

    // Odometry state
    std::mutex odom_mutex_;
    double ego_x_ = 0.0;
    double ego_y_ = 0.0;
    double ego_z_ = 0.0;
    double ego_yaw_ = 0.0;
    bool odom_received_ = false;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<LidarYoloFusionNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
