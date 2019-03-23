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
	glm::vec3 utm2ll(glm::vec3 p);
	glm::vec3 ll2utm(glm::vec3 p);
	static glm::vec3 px2uv(glm::vec3 p, cv::Rect satBB);
	static glm::vec3 uv2px(glm::vec3 p, cv::Rect satBB);
	static glm::vec2 px2uv(glm::vec2 p, cv::Rect satBB);
	static glm::vec2 uv2px(glm::vec2 p, cv::Rect satBB);

private:
	// Internal state
	int epsgCode;
	glm::vec3 origin;
	OGRSpatialReference srUTM, *srLL;
	OGRCoordinateTransformation* _utm2ll;
	OGRCoordinateTransformation* _ll2utm;
};

// A group of co-planar faces
struct SurfaceGroup {
	std::vector<int> faceIDs;	// List of face IDs in this group
	glm::mat4 xform;			// Rectify and place triangles onto atlas
	glm::ivec2 minBB;			// 2D bounding box (px, LL origin)
	glm::ivec2 maxBB;
	SurfaceGroup() : minBB(INT_MAX), maxBB(INT_MIN) {}
};

// Container for surface group statistics
struct GroupStats {
	int ta;		// Total area
	int tw;		// Total width
	int th;		// Total height
	int mw;		// Width of widest group
	int mh;		// Height of tallest group
	GroupStats() : ta(0), tw(0), th(0), mw(0), mh(0) {}
};

// Canvas object used to pack groups onto a texture of minimum area
class Canvas {
public:
	Canvas(glm::ivec2 size, GroupStats gs);

	int area() const { return sz.x * sz.y; }
	int width() const { return sz.x; }
	int height() const { return sz.y; }
	glm::ivec2 size() const { return sz; }
	int occupied(glm::uvec2 idx) const {
		return (idx.x < ws.size() && idx.y < hs.size()) ? occ[idx.x][idx.y] : 0; }
	bool packPossible() {
		if (area() < gs.ta) return false;	// Area too small
		if (sz.x < gs.mw) return false;		// Thinner than widest element
		if (sz.y < gs.mh) return false;		// Shorter than tallest element
		return true;
	}
	// Return the index of the tallest group in the rightmost column (or -1 if right column empty)
	int idxOfTallestRightGroup();
	// Return the index of the group that couldn't be places, if packing failed
	unsigned int idxOfUnplacedGroup() { return gi; }
	// Return the minimum height deficit
	int minHeightDeficit() { return mhd; }

	// Try to pack all groups onto the canvas
	bool pack(const std::vector<SurfaceGroup*>& groups, bool edit = false);
	// Remove any unused columns (and rows)
	void trim(bool rows = false);

private:
	glm::ivec2 sz;							// Total width and height (pixels)
	std::vector<std::vector<int>> occ;		// Which cells are occupied by which group (1-indexed)
	std::vector<int> ws;					// Width of each cell
	std::vector<int> hs;					// Height of each cell
	GroupStats gs;							// Stats about all the groups
	unsigned int gi;						// Index of group that couldn't be placed (if any)
	int mhd;								// Minimum height deficit when failed to place group in column

	// Split the cell at idx, given the size of the lower left sub-cell
	void split(glm::uvec2 idx, glm::ivec2 llsz);
};

// Returns the largest rectangle inscribed within regions of all non-zero pixels
cv::Rect findLargestRectangle(cv::Mat image);

// Histogram equalization with a mask image
cv::Mat equalizeHistMask(cv::Mat src, cv::Mat mask);

}

#endif
