#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <fstream>
#include "RtDB2.h"
// #include "def/def.hpp"
#include "RtDB_Barelang/rtdbBarelangDefinition.hpp"
#include "sl/Camera.hpp"
#include "def_zed.h"
#include "icecream.hpp"
#include "boost/program_options.hpp"
#include "boost/thread.hpp"
#include "filesystem"
#include "yaml-cpp/yaml.h"

#include "deploy/model.hpp"
#include "deploy/option.hpp"
#include "deploy/result.hpp"

// Using std and sl namespaces
using namespace std;
using namespace sl;
namespace po = boost::program_options;

boost::mutex mtx;
cv::VideoWriter videoWriter;
cv::VideoCapture cap;
bool refresh = false;

// barelang::TrackbarColour *Bola{nullptr};
// barelang::TrackbarColour *Robot{nullptr};
// barelang::TrackbarColour *Kipers{nullptr}; /// add

int robotId = 0;
std::string rtdb_config_file, network_name;
// int bolaHmin, bolaHmax, bolaSmin, bolaSmax, bolaVmin, bolaVmax;
// int kiperHmin, kiperHmax, kiperSmin, kiperSmax, kiperVmin, kiperVmax; /// add
void get_3d_coordinates(bbox_t &cur_box, const sl::Mat &pointCloud, sl::float4 &point);
bool running = true;
void sigHandler(int sig)
{
    running = false;
}
float getMedian(std::vector<float> &v)
{
    size_t n = v.size() / 2;
    std::nth_element(v.begin(), v.begin() + n, v.end());
    return v[n];
}

void get_3d_coordinates(bbox_t &cur_box, const sl::Mat &pointCloud, sl::float4 &point)
{
    bool valid_measure;
    int i, j;
    const unsigned int R_max_global = 10;

    const unsigned int obj_size = std::min(cur_box.w, cur_box.h);
    const unsigned int R_max = std::min(R_max_global, obj_size / 2);
    int center_i = cur_box.x + cur_box.w * 0.5f;
    int center_j = cur_box.y + cur_box.h * 0.5f;

    std::vector<float> x_vect, y_vect, z_vect;
    for (int R = 0; R < R_max; R++)
    {
        for (int y = -R; y <= R; y++)
        {
            for (int x = -R; x <= R; x++)
            {
                i = center_i + x;
                j = center_j + y;
                // sl::float4 out(NAN, NAN, NAN, NAN);
                sl::float4 out;
                if (i >= 0 && i < pointCloud.getWidth() && j >= 0 &&
                    j < pointCloud.getHeight())
                {
                    pointCloud.getValue(i, j, &out);
                }
                valid_measure = std::isfinite(out.z);
                if (valid_measure)
                {
                    x_vect.push_back(out.x);
                    y_vect.push_back(out.y);
                    z_vect.push_back(out.z);
                }
            }
        }
    }

    if (x_vect.size() > 0)
    {
        point.x = getMedian(x_vect);
        point.y = getMedian(y_vect);
        point.z = getMedian(z_vect);
    }
    else
    {
        point.x = NAN;
        point.y = NAN;
        point.z = NAN;
    }
}

bool req_record = false;

void visualize(cv::Mat &image, const deploy::DetectRes &result, const std::vector<std::string> &labels)
{
    for (size_t i = 0; i < result.num; ++i)
    {
        const auto &box = result.boxes[i];
        int cls = result.classes[i];
        float score = result.scores[i];
        const auto &label = labels[cls];
        std::string label_text = label + " " + cv::format("%.3f", score);

        // 绘制矩形和标签
        int base_line;
        cv::Size label_size = cv::getTextSize(label_text, cv::FONT_HERSHEY_SIMPLEX, 0.6, 1, &base_line);
        cv::rectangle(image, cv::Point(box.left, box.top), cv::Point(box.right, box.bottom), cv::Scalar(251, 81, 163), 2, cv::LINE_AA);
        cv::rectangle(image, cv::Point(box.left, box.top - label_size.height), cv::Point(box.left + label_size.width, box.top), cv::Scalar(125, 40, 81), -1);
        cv::putText(image, label_text, cv::Point(box.left, box.top), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(253, 168, 208), 1);
    }
}

void quatToEuler(float x, float y, float z, float w, float &roll, float &pitch, float &yaw)
{
    // roll (x-axis rotation)
    float sinr_cosp = 2 * (w * x + y * z);
    float cosr_cosp = 1 - 2 * (x * x + y * y);
    roll = atan2(sinr_cosp, cosr_cosp);

    // pitch (y-axis rotation)
    float sinp = 2 * (w * y - z * x);
    if (fabs(sinp) >= 1)
        pitch = copysign(M_PI / 2, sinp);
    else
        pitch = asin(sinp);

    // yaw (z-axis rotation)
    float siny_cosp = 2 * (w * z + x * y);
    float cosy_cosp = 1 - 2 * (y * y + z * z);
    yaw = atan2(siny_cosp, cosy_cosp);
}

int main(int argc, char **argv)
{
    int robotId = 0;
    sl::Camera zed;
    InitParameters init_parameters;
    po::options_description desc("BACA WOYY");
    desc.add_options()("help", "")("agent_id", po::value<int>(), "")("rtdb_config_file", po::value<std::string>(), "/opt/config/cfg.xml")("network_name", po::value<std::string>(), "");

    init_parameters.depth_mode = DEPTH_MODE::PERFORMANCE;
    init_parameters.coordinate_units = UNIT::CENTIMETER;
    init_parameters.camera_resolution = RESOLUTION::VGA;
    init_parameters.camera_fps = 30;
    init_parameters.coordinate_system = COORDINATE_SYSTEM::RIGHT_HANDED_Z_UP_X_FWD;
    init_parameters.depth_minimum_distance = 30;

    std::string engine_path = "/home/magenta/Documents/maen_boss/zed_ws/models/zed_obs.engine";
    std::string label_path = "/home/magenta/Documents/maen_boss/zed_ws/models/classes.txt";

    std::vector<std::string> labels;
    std::ifstream file(label_path);
    std::string label;

    while (std::getline(file, label))
    {
        labels.push_back(label);
    }

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help"))
    {
        IC(desc);
        return 1;
    }
    bool boleh_run = true;
    if (vm.count("agent_id"))
        robotId = vm["agent_id"].as<int>();
    else
    {
        std::cout << "--agent_id" << std::endl;
        boleh_run = false;
    }

    if (vm.count("rtdb_config_file"))
        rtdb_config_file = vm["rtdb_config_file"].as<std::string>();
    else
    {
        std::cout << "--rtdb_config_file" << std::endl;
        boleh_run = false;
    }
    if (vm.count("network_name"))
        network_name = vm["network_name"].as<std::string>();
    else
    {
        std::cout << "--network_name" << std::endl;
        boleh_run = false;
    }

    if (!boleh_run)
        return 1;

    RtDB2Context ctx = RtDB2Context::Builder(robotId, RtDB2ProcessType::comm).withConfigFileName(rtdb_config_file).withNetwork(network_name).build();
    RtDB2 rtdb(ctx);

    objPose ballPos, robotPos;

    ZED_t zed_data;

    auto returned_state = zed.open(init_parameters);
    if (returned_state != ERROR_CODE::SUCCESS)
    {
        std::cout << "Error " << returned_state << ", exit program." << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "################Start capturing################" << std::endl;

    // Activate Positional Tracking based on Camera
    // PositionalTrackingParameters trackingPosParameter;
    // trackingPosParameter.enable_pose_smoothing = true;
    // trackingPosParameter.mode = POSITIONAL_TRACKING_MODE::GEN_2;
    // ERROR_CODE errPosInit = zed.enablePositionalTracking(trackingPosParameter);
    // if (errPosInit != ERROR_CODE::SUCCESS)
    //     exit(-1);

    // sl::Pose positionalZED;
    sl::SensorsData imuZED;
    // umum::Point2D positionalResults;

    // Load YOLO model once sebelum loop
    deploy::InferOption option;
    option.enableSwapRB();
    auto model = std::make_unique<deploy::DetectModel>(engine_path, option);
    if (!model)
    {
        std::cerr << "Failed to load YOLO model." << std::endl;
        return EXIT_FAILURE;
    }

    while (true)
    {
        IMU imu_data;
        // if (zed.getPosition(positionalZED, REFERENCE_FRAME::WORLD) == POSITIONAL_TRACKING_STATE::OK)
        // {
            //Only Use x,y not z or imu bcs worst reading;
            // positionalResults.x = positionalZED.getTranslation().tx / 100.0;
            // positionalResults.y = positionalZED.getTranslation().ty / 100.0;
            // positionalResults.x = positionalZED.getTranslation().tx;
            // positionalResults.y = positionalZED.getTranslation().ty;

            // Get the zed IMU
            zed.getSensorsData(imuZED, TIME_REFERENCE::CURRENT);
            // std::cout << "IMU Orientation: {" << imuZED.imu.pose.getOrientation()  << "}\n";
            // float theta = imuZED.imu.pose.getOrientation().z;
            sl::Orientation q = imuZED.imu.pose.getOrientation();               
            sl::float3 euler;
            quatToEuler(q.x, q.y, q.z, q.w, euler.x, euler.y, euler.z);

            float roll = euler.x * 180.0f / M_PI;
            float pitch = euler.y * 180.0f / M_PI;
            float yaw = euler.z * 180.0f / M_PI;  
            // theta = fmod(theta, 2 * M_PI);
            // if (theta < 0)
            //     theta += M_PI * 2;
            if (yaw < 0)
                yaw += 360;
            imu_data.yaw = yaw;
            // imu_data.accelX = positionalResults.x;
            // imu_data.accelY = positionalResults.y;
            // printf("Position x: %.2f | y: %.2f  | theta %.2f\n", positionalResults.x , positionalResults.y, theta * 180.0 / M_PI);
            // printf("Position x: %.2f | y: %.2f  | theta %.2f\n", positionalResults.x , positionalResults.y, imu_data.yaw);

            // rtdb.put("ZED_POSITIONAL_TRACKING", &positionalResults);
            rtdb.put("DATA_IMU", &imu_data);
        // }

        if (zed.grab() == ERROR_CODE::SUCCESS)
        {
            sl::Mat image, point_cloud;
            zed.retrieveImage(image, VIEW::LEFT);
            zed.retrieveMeasure(point_cloud, MEASURE::XYZ, sl::MEM::CPU);
            // Pastikan gambar valid sebelum mengonversi
            // if (image.getHeight() == 0 || image.getWidth() == 0) {
            //     std::cerr << "Invalid image from ZED camera!" << std::endl;
            //     continue;
            // }

            // Konversi ke OpenCV Mat (BGRA)
            cv::Mat cvImage = cv::Mat((int)image.getHeight(), (int)image.getWidth(),
                                      CV_8UC4, image.getPtr<sl::uchar1>(sl::MEM::CPU));

            cv::Mat pointCloud = cv::Mat((int)point_cloud.getHeight(), (int)point_cloud.getWidth(), CV_8UC4, point_cloud.getPtr<sl::uchar1>(sl::MEM::CPU));

            // Periksa apakah cvImage berhasil dibuat
            if (cvImage.empty())
            {
                std::cerr << "Failed to convert ZED image to OpenCV Mat!" << std::endl;
                continue;
            }

            // Konversi dari BGRA ke BGR dengan variabel baru
            // cv::Mat cvImage;
            cv::cvtColor(cvImage, cvImage, cv::COLOR_BGRA2BGR);

            // Periksa apakah cvImage berhasil dibuat
            if (cvImage.empty())
            {
                std::cerr << "Failed to convert BGRA to BGR!" << std::endl;
                continue;
            }

            // Prediksi dengan YOLO
            deploy::Image img(cvImage.data, cvImage.cols, cvImage.rows);
            auto result = model->predict(img);
            visualize(cvImage, result, labels);

            int best_index_ball = -1;
            double best_confidence_ball = 0.6f;
            zed_data.obstacles_position_relative.clear(); // Clear Obstacle
            bool found_ball = false;

            ballPos.isDetected = false;
            for (size_t i = 0; i < result.num; ++i)
            {
                int classId = result.classes[i];

                if (labels[classId] == "Ball" && result.scores[i] > best_confidence_ball)
                {
                    best_confidence_ball = result.scores[i];
                    best_index_ball = i;
                    found_ball = true;
                }

                if (labels[classId] == "Obstacle")
                {
                    double thetaBot;
                    sl::float4 point;
                    const auto &box = result.boxes[i];

                    double x_obs = box.left;
                    double y_obs = box.top;
                    double w_obs = box.right;
                    double h_obs = box.bottom;

                    double centroid_botX = (x_obs + w_obs) / 2;
                    double centroid_botY = (y_obs + h_obs) / 2;

                    cv::line(cvImage, cv::Point(static_cast<int>(centroid_botX), static_cast<int>(centroid_botY)),
                             cv::Point(320, 480), cv::Scalar(0, 0, 0), 3);

                    point_cloud.getValue(centroid_botX, centroid_botY, &point);

                    if (std::isfinite(point.z))
                    {
                        robotPos.x = point.x + 13;
                        robotPos.y = -point.y + 3;
                        thetaBot = atan2(-robotPos.y, robotPos.x) * 180 / M_PI;
                        // if (thetaBot > 360)
                        //     thetaBot -= 360;
                        // if (thetaBot < 0)
                        //     thetaBot += 360;
                        // robotPos.z = thetaBot;
                        robotPos.z = (point.z) + 44;
                        robotPos.isDetected = true;
                        //                 if (IC(robotPos.isDetected && robotPos.z < 100))
                        // {

                        //     IC(robotPos.x, robotPos.y, robotPos.z);
                        // }
                    }

                    else
                    {
                        bbox_t bb;

                        bb.x = result.boxes[i].left;
                        bb.y = result.boxes[i].top;
                        bb.w = result.boxes[i].right;
                        bb.h = result.boxes[i].bottom;

                        get_3d_coordinates(bb, point_cloud, point);

                        if (std::isfinite(point.z))
                        {
                            robotPos.x = point.x + 13;
                            robotPos.y = -point.y + 3;
                            // thetaBot = atan2(-robotPos.y, robotPos.x) * 180 / M_PI  ;
                            // if (thetaBot > 360)
                            //     thetaBot -= 360;
                            // if (thetaBot < 0)
                            //     thetaBot += 360;
                            // robotPos.z = thetaBot;
                            robotPos.z = (point.z) + 44;
                            robotPos.isDetected = true;
                        }
                    }
                    // IC(robotPos.x, robotPos.y, robotPos.z);
                    // IC(result.classes[i], i);
                    zed_data.obstacles_position_relative.push_back(umum::Point2D(robotPos.x / 100.0, -robotPos.y / 100.0, robotPos.z / 100.0));
                }

                if (found_ball)
                {
                    double thetaBot;
                    sl::float4 point;
                    int i = best_index_ball;
                    double x_ball = result.boxes[i].left;
                    double y_ball = result.boxes[i].top;
                    double w_ball = result.boxes[i].right;
                    double h_ball = result.boxes[i].bottom;

                    float centroid_ballX = (x_ball + w_ball) / 2;
                    float centroid_ballY = (y_ball + h_ball) / 2;
                    // cv::circle(cvImage, cv::Point2d(centroid_ballX, centroid_ballY), 5, cv::Scalar(155, 0, 0), -1);
                    cv::line(cvImage, cv::Point(static_cast<int>(centroid_ballX), static_cast<int>(centroid_ballY)),
                             cv::Point(320, 480), cv::Scalar(0, 69, 255), 3);

                    point_cloud.getValue(centroid_ballX, centroid_ballY, &point);

                    if (std::isfinite(point.z))
                    {
                        IC(point.x, point.z, point.y);
                        ballPos.x = point.x + 13;
                        ballPos.y = -point.y + 3;

                        // thetaBall = atan2(-ballPos.y, ballPos.x) * 180.0 / M_PI;
                        // if (thetaBall > 360)
                        //     thetaBall -= 360;
                        // if (thetaBall < 0)
                        //     thetaBall += 360;
                        // ballPos.z = thetaBall;
                        ballPos.z = point.z + 44;
                        ballPos.isDetected = true;
                    }

                    // IC(ballPosX, ballPosY, ballPosZ);

                    else
                    {
                        bbox_t bb;
                        bb.x = result.boxes[i].left;
                        bb.y = result.boxes[i].top;
                        bb.w = result.boxes[i].right;
                        bb.h = result.boxes[i].bottom;

                        get_3d_coordinates(bb, point_cloud, point);

                        if (std::isfinite(point.z))
                        {
                            IC(point.x, point.z, point.y);

                            ballPos.x = point.x + 13;
                            ballPos.y = -point.y + 3;
                            // thetaBall = atan2(-ballPos.y, ballPos.x) * 180.0 / M_PI;
                            // if (thetaBall > 360)
                            //     thetaBall -= 360;
                            // if (thetaBall < 0)
                            //     thetaBall += 360;
                            ballPos.z = point.z + 44;
                            ballPos.isDetected = true;
                        }
                    }
                }
            }

            zed_data.ball_position_relative = {ballPos.x / 100., -ballPos.y / 100., ballPos.z / 100.};
            zed_data.nampak_bola = ballPos.isDetected;
            zed_data.height = ballPos.z / 100;

            // IC(zed_data.obstacles_position_relative.size());
            rtdb.put("ZED_FROM", &zed_data);
            // IC(ZED.Robot.x, ZED.Robot.y, ZED.Robot.z);
            // double dist = sqrt(pow(ZED.Robot.x, 2) + pow(ZED.Robot.y, 2));

            int zedReqAge = 0;
            bool zedReq = false;
            if (rtdb.get("ZED_STREAM_REQ", &zedReq, zedReqAge, robotId) == RTDB2_SUCCESS)
            {
                if (zedReqAge > 200)
                    zedReq = false;
            }
            else
                zedReq = false;

            cv::putText(cvImage, "Ball :", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
            cv::putText(cvImage, std::string("x : ") + std::to_string(zed_data.ball_position_relative.x), cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
            cv::putText(cvImage, std::string("y : ") + std::to_string(zed_data.ball_position_relative.y), cv::Point(10, 90), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
            // cv::putText(cvImage, std::string("z : ") + std::to_string(zed_data.height), cv::Point(10, 120), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
            cv::putText(cvImage, std::string("distance : ") + std::to_string(zed_data.ball_position_relative.dist()), cv::Point(10, 120), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
            if (zedReq)
            {

                Image_t zed_image_data;
                cv::Mat zedcv;
                // cv::cvtColor(cvImage, zedcv, cv::COLOR_BGRA2BGR);

                cv::resize(cvImage, zedcv, cv::Size(672 / 2, 376 / 2));

                zed_image_data.matrix = std::vector<unsigned char>(zedcv.data, zedcv.data + (zedcv.rows * zedcv.cols * zedcv.channels()));
                zed_image_data.rows = zedcv.rows;
                zed_image_data.cols = zedcv.cols;
                zed_image_data.type = zedcv.type();
                rtdb.put("ZED_STREAM", &zed_image_data);
                // cv::imshow("zed", frame);
            }

            // Tampilkan hasil
            cv::imshow("ZED_POV", cvImage);
            if (cv::waitKey(1) == 27)
                break; // Exit jika menekan ESC
        }
    }

    zed.close();
    cv::destroyAllWindows();

    return 0;
}
