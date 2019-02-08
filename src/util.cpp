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
glm::vec3 SpatXform::utm2px(glm::vec3 p, Satellite& sat, cv::Rect satBB) {
	// Add origin
	glm::dvec3 dp(p + origin);

	// UTM to lat/long
	utm2ll->Transform(1, &dp.x, &dp.y);

	// Lat/long to pixels
	int succ;
	GDALRPCTransform(sat.rpcXformer, FALSE, 1, &dp.x, &dp.y, &dp.z, &succ);

	// Relative to bounding box
	dp.x -= satBB.tl().x;
	dp.y -= satBB.tl().y;

	return glm::vec3(dp);
}

// Transform from pixels to UTM
glm::vec3 SpatXform::px2utm(glm::vec3 p, Satellite& sat, cv::Rect satBB) {
	glm::dvec3 dp(p);

	// Relative to bounding box
	dp.x += satBB.tl().x;
	dp.y += satBB.tl().y;

	// Pixels to lat/long
	int succ;
	GDALRPCTransform(sat.rpcXformer, TRUE, 1, &dp.x, &dp.y, &dp.z, &succ);

	// Lat/long to UTM
	ll2utm->Transform(1, &dp.x, &dp.y);

	// Subtract building origin
	dp = dp - glm::dvec3(origin);

	return glm::vec3(p);
}

glm::vec3 SpatXform::utm2uv(glm::vec3 p, Satellite& sat, cv::Rect satBB) {
	// Use image size if no rect passed
	if (satBB == cv::Rect())
		satBB = cv::Rect(0, 0, sat.satImg.cols, sat.satImg.rows);

	// Convert to px, then to uv
	return px2uv(utm2px(p, sat, satBB), satBB);
}
glm::vec3 SpatXform::uv2utm(glm::vec3 p, Satellite& sat, cv::Rect satBB) {
	// Use image size if no rect passed
	if (satBB == cv::Rect())
		satBB = cv::Rect(0, 0, sat.satImg.cols, sat.satImg.rows);

	// Convert to px, then to utm
	return px2utm(uv2px(p, satBB), sat, satBB);
}
glm::vec3 SpatXform::px2uv(glm::vec3 p, cv::Rect satBB) {
	// Do nothing if invalid rect passed
	if (satBB.width <= 0 || satBB.height <= 0)
		return p;

	// Convert to uv space and return
	glm::vec3 uv = p;
	uv.x = p.x / satBB.width;
	uv.y = 1.0 - p.y / satBB.height;
	return uv;
}
glm::vec3 SpatXform::uv2px(glm::vec3 p, cv::Rect satBB) {
	// Do nothing if invalid rect passed
	if (satBB.width <= 0 || satBB.height <= 0)
		return p;

	// Convert to px space and return
	glm::vec3 px = p;
	px.x = p.x * satBB.width;
	px.y = (1.0 - p.y) * satBB.height;
	return px;
}

}
