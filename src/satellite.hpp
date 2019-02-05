#ifndef SATELLITE_HPP
#define SATELLITE_HPP

#include <string>
#include <opencv2/opencv.hpp>
#include <gdal/gdal_priv.h>
#include <gdal/gdal_alg.h>
#include <glm/glm.hpp>

class Satellite {
public:
	cv::Mat satImg;
	GDALRPCInfo rpcInfo;
	void* rpcXformer;
	cv::Rect bb;
	std::string name;
	glm::vec2 projUp;

public:
	Satellite(std::string filename);
	~Satellite();
	// Disable copy
	Satellite(const Satellite& other) = delete;
	Satellite(Satellite&& other);
	// Disable any kind of assignment
	Satellite& operator=(const Satellite& other) = delete;
	Satellite& operator=(Satellite&& other) = delete;

	void calcBB(std::vector<cv::Point2f> allPts, int border = 50);
};

//cv::Point3d utm2px(cv::Point3d p, BuildingMetadata& bm, Satellite& sat);
//cv::Point3d px2utm(cv::Point3d p, BuildingMetadata& bm, Satellite& sat);

#endif
