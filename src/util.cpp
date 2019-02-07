#include "util.hpp"

namespace util {

SpatXform::SpatXform(uint32_t epsgCode, glm::vec3 origin) :
	epsgCode(epsgCode), origin(origin) {
	// Init transformation structures
	srUTM.importFromEPSG(epsgCode);
	srLL = srUTM.CloneGeogCS();
	utm2ll = OGRCreateCoordinateTransformation(&srUTM, srLL);
	ll2utm = OGRCreateCoordinateTransformation(srLL, &srUTM);
}

// Transform from UTM to pixels
glm::dvec3 SpatXform::utm2px(glm::dvec3 p, Satellite& sat, cv::Rect satBB) {
	// Add origin
	p = p + glm::dvec3(origin);

	// UTM to lat/long
	utm2ll->Transform(1, &p.x, &p.y);

	// Lat/long to pixels
	int succ;
	GDALRPCTransform(sat.rpcXformer, FALSE, 1, &p.x, &p.y, &p.z, &succ);

	// Relative to bounding box
	if (satBB != cv::Rect()) {
		p.x -= satBB.tl().x;
		p.y -= satBB.tl().y;
	}

	return p;
}

// Transform from pixels to UTM
glm::dvec3 SpatXform::px2utm(glm::dvec3 p, Satellite& sat, cv::Rect satBB) {

	// Relative to bounding box
	if (satBB != cv::Rect()) {
		p.x += satBB.tl().x;
		p.y += satBB.tl().y;
	}

	// Pixels to lat/long
	int succ;
	GDALRPCTransform(sat.rpcXformer, TRUE, 1, &p.x, &p.y, &p.z, &succ);

	// Lat/long to UTM
	ll2utm->Transform(1, &p.x, &p.y);

	// Subtract building origin
	p = p - glm::dvec3(origin);

	return p;
}

}
