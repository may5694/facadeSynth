#include "dn_predict.hpp"

#include <torch/script.h> // One-stop header.

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <memory>

int const max_BINARY_value = 255;
int const height = 224; // DNN image height
int const width = 224; // DNN image width
int const num_paras = 4; // number of paras

std::vector<double> dn_predict(cv::Mat src, std::string network_path) {

	// Deserialize the ScriptModule from a file using torch::jit::load().
	std::shared_ptr<torch::jit::script::Module> module = torch::jit::load(network_path);

	module->to(at::kCUDA);

	assert(module != nullptr);
	std::cout << "ok\n";

	cv::Mat dst_ehist, dst_classify;
	// Convert to grayscale
	cvtColor(src, src, CV_BGR2GRAY);
	// Apply Histogram Equalization
	equalizeHist(src, dst_ehist);
	// threshold classification
	int threshold = 90;
	cv::threshold(dst_ehist, dst_classify, threshold, max_BINARY_value, cv::THRESH_BINARY);
	// generate input image for DNN
	cv::Scalar bg_color(255, 255, 255); // white back ground
	cv::Scalar window_color(0, 0, 0); // black for windows
	cv::Mat scale_img;
	cv::resize(dst_classify, scale_img, cv::Size(width, height));
	std::vector<std::vector<cv::Point> > contours;
	std::vector<cv::Vec4i> hierarchy;
	cv::findContours(scale_img, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));

	std::vector<cv::Rect> boundRect(contours.size());
	for (int i = 0; i < contours.size(); i++)
	{
		boundRect[i] = cv::boundingRect(cv::Mat(contours[i]));
	}
	cv::Mat dnn_img(scale_img.size(), CV_8UC3, bg_color);
	for (int i = 0; i< contours.size(); i++)
	{
		if (hierarchy[i][2] != -1) continue;
		cv::rectangle(dnn_img, cv::Point(boundRect[i].tl().x + 1, boundRect[i].tl().y + 1), cv::Point(boundRect[i].br().x, boundRect[i].br().y), window_color, -1);
	}
	cv::cvtColor(dnn_img, dnn_img, CV_BGR2RGB);
	cv::Mat img_float;
	dnn_img.convertTo(img_float, CV_32F, 1.0 / 255);
	auto img_tensor = torch::from_blob(img_float.data, { 1, 224, 224, 3 }).to(torch::kCUDA);
	img_tensor = img_tensor.permute({ 0, 3, 1, 2 });
	img_tensor[0][0] = img_tensor[0][0].sub(0.485).div(0.229);
	img_tensor[0][1] = img_tensor[0][1].sub(0.456).div(0.224);
	img_tensor[0][2] = img_tensor[0][2].sub(0.406).div(0.225);

	std::vector<torch::jit::IValue> inputs;
	inputs.push_back(img_tensor);
	torch::Tensor out_tensor = module->forward(inputs).toTensor();
	std::cout << out_tensor.slice(1, 0, num_paras) << std::endl;
	std::vector<double> paras;
	for (int i = 0; i < num_paras; i++) {
		paras.push_back(out_tensor.slice(1, i, i+1).item<float>());
	}

	return paras;
}

cv::Mat generateFacadeSynImage(std::vector<double> params) {

	// predict img by DNN
	std::pair<int, int> imageRows(5, 20); // configs from synthetic images
	std::pair<int, int> imageCols(10, 20);
	int img_rows = round(params[0] * (imageRows.second - imageRows.first) + imageRows.first);
	int img_cols = round(params[1] * (imageCols.second - imageCols.first) + imageCols.first);
	int img_groups = 1;
	double relative_width = params[2];
	double relative_height = params[3];

	cv::Scalar bg_color(255, 255, 255); // white back ground
	cv::Scalar window_color(0, 0, 0); // black for windows
	int NR = img_rows;
	int NC = img_cols;
	int NG = img_groups;
	double ratioWidth = relative_width;
	double ratioHeight = relative_height;
	int thickness = -1;
	cv::Mat result(height, width, CV_8UC3, bg_color);
	double FH = height * 1.0 / NR;
	double FW = width * 1.0 / NC;
	double WH = FH * ratioHeight;
	double WW = FW * ratioWidth;
	std::cout << "NR is " << NR << std::endl;
	std::cout << "NC is " << NC << std::endl;
	std::cout << "FH is " << FH << std::endl;
	std::cout << "FW is " << FW << std::endl;
	std::cout << "ratioWidth is " << ratioWidth << std::endl;
	std::cout << "ratioHeight is " << ratioHeight << std::endl;
	std::cout << "WH is " << WH << std::endl;
	std::cout << "WW is " << WW << std::endl;
	// draw facade image
	if (NG == 1) {
		for (int i = 0; i < NR; ++i) {
			for (int j = 0; j < NC; ++j) {
				float x1 = (FW - WW) * 0.5 + FW * j;
				float y1 = (FH - WH) * 0.5 + FH * i;
				float x2 = x1 + WW;
				float y2 = y1 + WH;
				cv::rectangle(result, cv::Point(std::round(x1), std::round(y1)), cv::Point(std::round(x2), std::round(y2)), window_color, thickness);
			}
		}
	}
	else {
		double GFW = WW / NG;
		double GWW = WW / NG - 2;
		for (int i = 0; i < NR; ++i) {
			for (int j = 0; j < NC; ++j) {
				float x1 = (FW - WW) * 0.5 + FW * j;
				float y1 = (FH - WH) * 0.5 + FH * i;
				for (int k = 0; k < NG; k++) {
					float g_x1 = x1 + GFW * k;
					float g_y1 = y1;
					float g_x2 = g_x1 + GWW;
					float g_y2 = g_y1 + WH;

					cv::rectangle(result, cv::Point(std::round(g_x1), std::round(g_y1)), cv::Point(std::round(g_x2), std::round(g_y2)), window_color, thickness);
				}
			}
		}
	}
	return result;
}
