#ifndef UTIL_HPP
#define UTIL_HPP

#include <glm/glm.hpp>
#include <gdal/ogr_spatialref.h>
#include "satellite.hpp"

namespace util {

// Transform between UTM and pixel spaces
class SpatXform {
public:
	SpatXform(uint32_t epsgCode, glm::vec3 origin);

	// Transform methods
	glm::dvec3 utm2px(glm::dvec3 p, Satellite& sat, cv::Rect satBB = {});
	glm::dvec3 px2utm(glm::dvec3 p, Satellite& sat, cv::Rect satBB = {});

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
