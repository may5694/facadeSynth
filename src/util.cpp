#include "util.hpp"
using namespace std;

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


// Canvas constructor
Canvas::Canvas(glm::ivec2 size, GroupStats gs) : sz(size), gs(gs), gi(0) {
	assert(sz.x > 0 && sz.y > 0);
	occ.push_back(vector<int>(1, 0));
	ws.push_back(sz.x);
	hs.push_back(sz.y);
}


// Return the index of the tallest group in the rightmost column (or -1 if right column empty)
int Canvas::idxOfTallestRightGroup() {
	int minOcc = -1;	// Start by assuming nothing in right column
	for (size_t r = 0; r < hs.size(); r++) {
		int o = occ[ws.size() - 1][r];
		if (o) {
			// If this is the first occupied cell we've found, set it
			if (minOcc == -1)
				minOcc = o - 1;
			// If this is not the first occupied cell, keep the minimum index
			else
				minOcc = min(minOcc, o - 1);
		}
	}
	return minOcc;
}

// Try to pack all groups onto the canvas
bool Canvas::pack(const vector<SurfaceGroup*>& groups, bool edit) {
	if (!packPossible()) return false;

	// Try to fit each group onto the canvas, without overlaps
	mhd = INT_MAX;
	for (gi = 0; gi < groups.size(); gi++) {
		SurfaceGroup* g = groups[gi];
		bool placed = false;
		glm::uvec2 idx(0, 0);
		// Look at each cell, from left to right, bottom to top
		for (idx.x = 0; idx.x < ws.size(); idx.x++) {
			for (idx.y = 0; idx.y < hs.size(); idx.y++) {
				// If cell is free
				if (!occ[idx.x][idx.y]) {
					glm::ivec2 gSz = g->maxBB - g->minBB;
					glm::ivec2 cSz(ws[idx.x], hs[idx.y]);
					glm::uvec2 idx2 = idx + glm::uvec2(1, 1);
					// Find out how many cells we need to check to the right
					while (gSz.x > cSz.x && idx2.x < ws.size()) {
						cSz.x += ws[idx2.x++];
					}
					// Can't place this group anywhere -- too far right, abort the whole thing
					if (gSz.x > cSz.x) return false;
					// Find out how many cells we need to check above
					while (gSz.y > cSz.y && idx2.y < hs.size()) {
						cSz.y += hs[idx2.y++];
					}
					// Can't place this group in this column, too far up -- but keep going to the right
					if (gSz.y > cSz.y) {
						// Keep track of minimum height deficit
						mhd = min(mhd, gSz.y - cSz.y);
						continue;
					}

					// Check all the cells needed to place this group
					bool fits = true;
					for (unsigned int c = idx.x; c < idx2.x; c++) {
						for (unsigned int r = idx.y; r < idx2.y; r++) {
							if (occ[c][r]) {
								fits = false;
								break;
							}
						}
						if (!fits) break;
					}
					// Group doesn't fit here, keep looking
					if (!fits) continue;

					// Group does fit here, split the upper-right cell to fit
					split(idx2 - glm::uvec2(1, 1), gSz - (cSz - glm::ivec2(ws[idx2.x - 1], hs[idx2.y - 1])));
					// Mark all cells as occupied
					for (unsigned int c = idx.x; c < idx2.x; c++) {
						for (unsigned int r = idx.y; r < idx2.y; r++) {
							occ[c][r] = gi + 1;
						}
					}
					placed = true;
					// Get the position of the placed group
					glm::ivec2 pos(0);
					for (unsigned int c = 0; c < idx.x; c++) pos.x += ws[c];
					for (unsigned int r = 0; r < idx.y; r++) pos.y += hs[r];
					if (edit) {
						// Update the group's transformation
						glm::mat4 xlate = glm::mat4(1.0);
						xlate[3] = glm::vec4(-g->minBB + pos, 0.0, 1.0);
						g->xform = xlate * g->xform;
						// Update the group's bounding box
						g->maxBB = g->maxBB - g->minBB + pos;
						g->minBB = pos;
					}
					break;
				}
			}
			if (placed) break;
		}
		// Couldn't find a place for this group, stop trying
		if (!placed) return false;
	}

	// All groups were packed!
	trim();
	return true;
}

// Remove any unused columns (and rows)
void Canvas::trim(bool rows) {
	// Look for unoccupied columns, starting from the right
	for (size_t c = ws.size() - 1; c > 0; c--) {
		bool unocc = true;
		// Check every row in this column
		for (size_t r = 0; r < hs.size(); r++) {
			if (occ[c][r]) {
				unocc = false;
				break;
			}
		}
		// Column completely unoccupied, remove it
		if (unocc) {
			occ.pop_back();
			sz.x -= ws[c];
			ws.pop_back();
			// Column occupied somewhere, don't check any others
		}
		else {
			break;
		}
	}

	if (rows) {
		// Look for unoccupied rows, starting from the top
		for (size_t r = hs.size() - 1; r > 0; r--) {
			bool unocc = true;
			// Check every column in this row
			for (size_t c = 0; c < ws.size(); c++) {
				if (occ[c][r]) {
					unocc = false;
					break;
				}
			}
			// Row completely unoccupied, remove it
			if (unocc) {
				for (auto& col : occ) {
					col.pop_back();
				}
				sz.y -= hs[r];
				hs.pop_back();
				// Row occupied somewhere, don't check any others
			}
			else {
				break;
			}
		}
	}
}

// Split the cell at idx, given the size of the lower left sub-cell
void Canvas::split(glm::uvec2 idx, glm::ivec2 llsz) {
	assert(idx.x < ws.size() && idx.y < hs.size());
	// Split a row
	if (llsz.y > 0 && llsz.y < hs[idx.y]) {
		for (auto& col : occ) {
			// Insert a copy of the cell directly below it in each column
			col.insert(col.begin() + idx.y, col[idx.y]);
		}
		// Subtract the height of the lower row
		hs[idx.y] -= llsz.y;
		// Insert the height of the lower row
		hs.insert(hs.begin() + idx.y, llsz.y);
	}
	// Split a column
	if (llsz.x > 0 && llsz.x < ws[idx.x]) {
		// Insert a copy of the column directly before it
		occ.insert(occ.begin() + idx.x, occ[idx.x]);
		// Subtract the width of the left column
		ws[idx.x] -= llsz.x;
		// Insert the width of the left column
		ws.insert(ws.begin() + idx.x, llsz.x);
	}
}


}
