#ifndef DN_PREDICT_HPP
#define DN_PREDICT_HPP

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "rapidjson/document.h"
#include "rapidjson/writer.h"
#include "rapidjson/prettywriter.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/filereadstream.h"
#include "rapidjson/filewritestream.h"


int const cluster_number = 2;

void dn_predict(std::string metajson, std::string modeljson);
double readNumber(const rapidjson::Value& node, const char* key, double default_value);
std::vector<double> read1DArray(const rapidjson::Value& node, const char* key);
bool readBoolValue(const rapidjson::Value& node, const char* key, bool default_value);
std::string readStringValue(const rapidjson::Value& node, const char* key);

/**** steps *****/
bool chipping(std::string metajson, std::string modeljson, cv::Mat& croppedImage, bool bDebug);
cv::Mat crop_chip(cv::Mat src_chip, int type, bool bground, std::vector<double> facChip_size, double target_width, double target_height);
cv::Mat adjust_chip(cv::Mat src_chip, cv::Mat chip, int type, bool bground, std::vector<double> facChip_size, double target_width, double target_height);
cv::Mat facade_clustering_kkmeans(cv::Mat src_img, int clusters);

bool segment_chip(cv::Mat croppedImage, cv::Mat& dnn_img, std::string metajson, std::string modeljson, bool bDebug);
float findSkewAngle(cv::Mat src_img);
cv::Mat cleanAlignedImage(cv::Mat src, float threshold);
void writeBackAvgColors(std::string metajson, cv::Scalar bg_avg_color, cv::Scalar win_avg_color);

std::vector<double> feedDnn(cv::Mat dnn_img, std::string metajson, std::string modeljson, bool bDebug);
std::vector<double> grammar1(std::string modeljson, std::vector<double> paras, bool bDebug);
std::vector<double> grammar2(std::string modeljson, std::vector<double> paras, bool bDebug);
std::vector<double> grammar3(std::string modeljson, std::vector<double> paras, bool bDebug);
std::vector<double> grammar4(std::string modeljson, std::vector<double> paras, bool bDebug);
std::vector<double> grammar5(std::string modeljson, std::vector<double> paras, bool bDebug);
std::vector<double> grammar6(std::string modeljson, std::vector<double> paras, bool bDebug);

#endif
