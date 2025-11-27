#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <chrono> // 시간 측정을 위한 헤더 추가
#include <cmath>
#include <algorithm>
#include <map>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "tf2_ros/static_transform_broadcaster.h"
#include "cv_bridge/cv_bridge.h"
#include "pcl_conversions/pcl_conversions.h"
#include "pcl/point_cloud.h"
#include "pcl/point_types.h"
#include "message_filters/subscriber.h"
#include "message_filters/synchronizer.h"
#include "message_filters/sync_policies/approximate_time.h"

#include <Eigen/Dense>

#include <cuda_runtime_api.h>
#include <NvInfer.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;
using namespace std::chrono; // chrono 네임스페이스 사용

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

class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) cout << "[TRT] " << msg << std::endl;
    }
} gLogger;

class YoloDetector {
public:
    YoloDetector(const std::string& engine_path, float conf_thres, float iou_thres)
        : conf_thres_(conf_thres), iou_thres_(iou_thres)
    {
        // ... (엔진 로딩 및 스트림 생성 부분은 기존과 동일) ...
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

        detector_ = std::make_shared<YoloDetector>(model_path, conf, iou);

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
            
            // 1. 라이다 데이터 변환
            pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
            pcl::fromROSMsg(*lidar_msg, *cloud);
            if (cloud->empty()) return;

            std::vector<bool> is_object_point(cloud->points.size(), false);

            std::vector<sensor_msgs::msg::Image::ConstSharedPtr> img_msgs = {front_msg, left_msg, right_msg, back_msg};
            std::vector<cv::Mat> cv_imgs(4);

            for (int i = 0; i < 4; i++) {
                try {
                    cv_imgs[i] = cv_bridge::toCvCopy(img_msgs[i], "bgr8")->image.clone();
                }
                catch (cv_bridge::Exception& e) {
                    RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
                    return;
                }
            }

            std::vector<Eigen::Vector3f> all_centers;
            std::vector<std::string> all_labels;
            std::vector<float> all_dists;
            std::vector<int> target_classes = {0, 2, 5, 7};


            auto start_infer = high_resolution_clock::now();
            for (int i = 0; i < 4; i++) {
                std::vector<Detection> results = detector_->detect(cv_imgs[i]);
                Eigen::Matrix4f T_lidar2cam = extrinsic_list_[i];
                
                std::vector<cv::Point> valid_uvs;
                std::vector<int> valid_indices;

                for (size_t k = 0; k < cloud->points.size(); k++) {
                    const auto& pt = cloud->points[k];
                    Eigen::Vector4f p_lidar(pt.x, pt.y, pt.z, 1.0f);
                    Eigen::Vector4f p_cam = T_lidar2cam * p_lidar;

                    if (p_cam(2) <= 0) continue; 

                    Eigen::Vector3f proj = K_ * p_cam.head<3>();
                    if (proj(2) == 0) continue;

                    int u = static_cast<int>(proj(0) / proj(2));
                    int v = static_cast<int>(proj(1) / proj(2));

                    if (u >= 0 && u < cv_imgs[i].cols && v >= 0 && v < cv_imgs[i].rows) {
                        valid_uvs.emplace_back(u, v);
                        valid_indices.push_back(k);
                    }
                }

                for (auto& det : results) {
                    if (det.box.area() <= 0) continue;

                    cv::rectangle(cv_imgs[i], det.box, cv::Scalar(0, 255, 0), 2);
                    std::string label = class_names_[det.class_id];
                    cv::putText(cv_imgs[i], label, det.box.tl(), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);

                    if (!det.mask.empty()) {
                        cv::Mat colored_mask;
                        cv::cvtColor(det.mask, colored_mask, cv::COLOR_GRAY2BGR);
                        colored_mask.setTo(cv::Scalar(0, 0, 255), det.mask > 0); 
                        
                        cv::Rect valid_box = det.box & cv::Rect(0, 0, cv_imgs[i].cols, cv_imgs[i].rows);
                        if (valid_box.area() > 0) {
                            cv::Mat roi = cv_imgs[i](valid_box);
                            cv::Mat mask_roi = colored_mask(cv::Rect(0, 0, valid_box.width, valid_box.height));
                            cv::addWeighted(roi, 1.0, mask_roi, 0.4, 0.0, roi);
                        }
                    }

                    bool is_target = false;
                    for(int t_id : target_classes) {
                        if(det.class_id == t_id) { is_target = true; break; }
                    }

                    Eigen::Vector3f sum_points(0, 0, 0);
                    int count_points = 0;
                    float min_dist = 9999.0f;

                    if (is_target && !det.mask.empty()) {
                        
                        // ==========================================
                        // [추가된 부분] 마스크 축소 (0.8배) 로직
                        // ==========================================
                        cv::Mat final_check_mask;
                        float scale_ratio = 0.8f; // 80% 크기로 축소

                        int new_w = static_cast<int>(det.mask.cols * scale_ratio);
                        int new_h = static_cast<int>(det.mask.rows * scale_ratio);

                        // 최소 크기 보호
                        if (new_w > 0 && new_h > 0) {
                            // 1. 검은색 배경 생성 (원본 마스크 크기)
                            final_check_mask = cv::Mat::zeros(det.mask.size(), det.mask.type());
                            
                            // 2. 마스크 내용을 0.8배로 리사이즈
                            cv::Mat resized_content;
                            cv::resize(det.mask, resized_content, cv::Size(new_w, new_h), 0, 0, cv::INTER_NEAREST);
                            
                            // 3. 중앙에 배치 (Padding 효과)
                            int offset_x = (det.mask.cols - new_w) / 2;
                            int offset_y = (det.mask.rows - new_h) / 2;
                            
                            resized_content.copyTo(final_check_mask(cv::Rect(offset_x, offset_y, new_w, new_h)));
                        } else {
                            // 너무 작아서 축소가 불가능하면 원본 사용
                            final_check_mask = det.mask; 
                        }
                        // ==========================================

                        for (size_t k = 0; k < valid_uvs.size(); ++k) {
                            cv::Point pt = valid_uvs[k];

                            if (det.box.contains(pt)) {
                                int mx = pt.x - det.box.x;
                                int my = pt.y - det.box.y;

                                if (mx >= 0 && mx < final_check_mask.cols && my >= 0 && my < final_check_mask.rows) {
                                    // [수정] 원본 det.mask 대신 축소된 final_check_mask 사용
                                    if (final_check_mask.at<uint8_t>(my, mx) > 0) {
                                        int original_idx = valid_indices[k];
                                        
                                        is_object_point[original_idx] = true;

                                        const auto& p3d = cloud->points[original_idx];
                                        Eigen::Vector3f p_vec(p3d.x, p3d.y, p3d.z);
                                        
                                        float dist = p_vec.norm();
                                        if (dist < min_dist) min_dist = dist;

                                        sum_points += p_vec;
                                        count_points++;
                                    }
                                }
                            }
                        }

                        if (count_points > 0) {
                            Eigen::Vector3f center = sum_points / static_cast<float>(count_points);
                            all_centers.push_back(center);
                            all_labels.push_back(label);
                            all_dists.push_back(min_dist);
                        }
                    }
                }

                sensor_msgs::msg::Image::SharedPtr out_msg = cv_bridge::CvImage(img_msgs[i]->header, "bgr8", cv_imgs[i]).toImageMsg();
                overlay_pubs_[i]->publish(*out_msg);
            }

            auto end_infer = high_resolution_clock::now();
            auto duration_infer = duration_cast<milliseconds>(end_infer - start_infer);
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

            if (!all_centers.empty()) {
                cv::Mat top_view = build_topview(all_centers, all_labels, all_dists);
                sensor_msgs::msg::Image::SharedPtr top_msg = cv_bridge::CvImage(front_msg->header, "bgr8", top_view).toImageMsg();
                pub_topview_->publish(*top_msg);
            }

            auto end_total = high_resolution_clock::now();
            auto duration_total = duration_cast<milliseconds>(end_total - start_total);
            // [로그 출력] 태그: [CPP_PERF]
            RCLCPP_INFO(this->get_logger(), "[CPP_PERF] Inference: %ld ms, Total: %ld ms", 
            duration_infer.count(), duration_total.count());

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

    std::shared_ptr<YoloDetector> detector_;
    std::shared_ptr<tf2_ros::StaticTransformBroadcaster> tf_broadcaster_;
    message_filters::Subscriber<sensor_msgs::msg::PointCloud2> sub_lidar_;
    message_filters::Subscriber<sensor_msgs::msg::Image> sub_front_, sub_left_, sub_right_, sub_back_;
    std::shared_ptr<Sync> sync_;

    std::vector<rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr> overlay_pubs_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_obj_pc_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_topview_;

    std::vector<Eigen::Matrix4f> extrinsic_list_;
    std::vector<Eigen::Matrix4f> cam2lidar_list_;
    Eigen::Matrix3f K_;
    std::map<int, std::string> class_names_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<LidarYoloFusionNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
