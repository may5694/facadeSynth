#ifndef UTIL_HPP
#define UTIL_HPP

#include <glm/glm.hpp>
#include <gdal/ogr_spatialref.h>
#include "satellite.hpp"

namespace util {

// Transform between UTM, pixel, and UV spaces
class SpatXform {
public:
	SpatXform(uint32_t epsgCode, glm::vec3 origin);

	// Transform methods
	glm::vec3 utm2px(glm::vec3 p, Satellite& sat, cv::Rect satBB = {});
	glm::vec3 px2utm(glm::vec3 p, Satellite& sat, cv::Rect satBB = {});
	glm::vec3 utm2uv(glm::vec3 p, Satellite& sat, cv::Rect satBB = {});
	glm::vec3 uv2utm(glm::vec3 p, Satellite& sat, cv::Rect satBB = {});
	static glm::vec3 px2uv(glm::vec3 p, cv::Rect satBB);
	static glm::vec3 uv2px(glm::vec3 p, cv::Rect satBB);

private:
	// Internal state
	int epsgCode;
	glm::vec3 origin;
	OGRSpatialReference srUTM, *srLL;
	OGRCoordinateTransformation* utm2ll;
	OGRCoordinateTransformation* ll2utm;
};

}

#endif
