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

void dn_predict(std::string metajson, std::string modeljson);
int find_threshold(cv::Mat src, bool bground);
double readNumber(const rapidjson::Value& node, const char* key, double default_value);
std::vector<double> read1DArray(const rapidjson::Value& node, const char* key);
bool readBoolValue(const rapidjson::Value& node, const char* key, bool default_value);
std::string readStringValue(const rapidjson::Value& node, const char* key);

#endif
