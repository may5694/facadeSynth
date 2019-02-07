#ifndef SATELLITE_HPP
#define SATELLITE_HPP

#include <string>
#include <opencv2/opencv.hpp>
#include <gdal/gdal_priv.h>
#include <gdal/gdal_alg.h>
#include <glm/glm.hpp>

// Holds a satellite dataset with associated RPC info
class Satellite {
public:
	cv::Mat satImg;			// The satellite image as an OpenCV Mat
	GDALRPCInfo rpcInfo;	// RPC info
	void* rpcXformer;		// RPC transformer object
	std::string name;		// Name of satellite dataset
	glm::vec2 projUp;		// Projected "up" vector

public:
	Satellite(std::string filename);
	~Satellite();
	// Disable copy
	Satellite(const Satellite& other) = delete;
	Satellite(Satellite&& other);
	// Disable any kind of assignment
	Satellite& operator=(const Satellite& other) = delete;
	Satellite& operator=(Satellite&& other) = delete;

//	void calcBB(std::vector<cv::Point2f> allPts, int border = 50);
};

#endif
