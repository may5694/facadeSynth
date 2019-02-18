#ifndef DN_PREDICT_HPP
#define DN_PREDICT_HPP

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <rapidjson/document.h>

rapidjson::Document dn_predict(cv::Mat image, rapidjson::Document meta, std::string network_path);
//cv::Mat generateFacadeSynImage(std::vector<double> params);

#endif
