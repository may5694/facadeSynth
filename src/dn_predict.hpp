#ifndef DN_PREDICT_HPP
#define DN_PREDICT_HPP

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

std::vector<double> dn_predict(cv::Mat src, std::string network_path);
cv::Mat generateFacadeSynImage(std::vector<double> params);

#endif
