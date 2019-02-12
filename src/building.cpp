#include "building.hpp"
#include "json.hpp"
#include "util.hpp"
#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"
using namespace std;
using json = nlohmann::json;
namespace fs = std::experimental::filesystem;

// Generate building data from input directory, and save it to data directory
void Building::generate(fs::path inputDir, fs::path dataDir, map<string, Satellite>& sats,
	string region, string cluster, string model) {
	// Clear any existing contents
	clear();

	// Check for satellite images
	if (sats.empty())
		throw runtime_error("No satellite images!");

	// Check for input directory existence
	fs::path inputClusterDir = inputDir / region / "BuildingClusters" / cluster;
	if (!fs::exists(inputClusterDir)) {
		stringstream ss;
		ss << "Couldn't find cluster \"" << cluster << "\", region \""
			<< region << "\" in " << inputClusterDir;
		throw runtime_error(ss.str());
	}
	fs::path inputModelDir = inputClusterDir / "ModelsOutput" / model;
	if (!fs::exists(inputModelDir)) {
		stringstream ss;
		ss << "Couldn't find model \"" << model << "\" in " << inputModelDir;
		throw runtime_error(ss.str());
	}

	// Check for data directory existence
	if (!fs::exists(dataDir)) {
		stringstream ss;
		ss << "Data directory " << dataDir << " does not exist!" << endl;
		throw runtime_error(ss.str());
	}

	this->region = region;
	this->cluster = cluster;
	this->model = model;

	// Read metadata
	genReadMetadata(inputClusterDir);

	// Generate geometry from input obj
	genGeometry(inputModelDir, sats);

	// Group faces into facades and generate atlas coordinates
	genFacades();

	// Write generated data to disk
	genWriteData(dataDir);
}

// Load building data from data directory
void Building::load(fs::path dataDir, string region, string cluster, string model) {
	// Clear any existing contents
	clear();

	// Check path to building data
	modelDir = dataDir / "regions" / region / cluster / model;
	if (!fs::exists(modelDir)) {
		stringstream ss;
		ss << "No data for model \"" << model << "\", cluster \"" << cluster
			<< "\", region \"" << region << "\" in " << modelDir;
		throw runtime_error(ss.str());
	}

	// Construct path to .obj file
	fs::path objPath = modelDir / (region + "_" + cluster + "_" + model + ".obj");
	// Construct path to .json file
	fs::path metaPath = objPath; metaPath.replace_extension(".json");

	// Load the geometry from the .obj file
	loadGeometry(objPath);
	loadMetadata(metaPath);

	// Check a few things
	if (this->region != region)
		throw runtime_error("Region mismatch - tried to load "
			+ region + " but got " + this->region);
	if (this->cluster != cluster)
		throw runtime_error("Cluster mismatch - tried to load "
			+ cluster + " but got " + this->cluster);
	if (this->model != model)
		throw runtime_error("Model mismatch - tried to load "
			+ model + " but got " + this->model);
	for (auto& s : satTCBufs)
		if (!satInfo.count(s.first))
			throw runtime_error("Satellite group " + s.first + " not found in satInfo!");
	for (auto& s : satInfo)
		if (!satTCBufs.count(s.first))
			throw runtime_error("Satellite info " + s.first + " not found in texcoords!");
}

// Release all memory and return to empty state
void Building::clear() {
	modelDir = fs::path();
	// Clear geometry buffers
	posBuf.clear();
	normBuf.clear();
	atlasTCBuf.clear();
	satTCBufs.clear();
	indexBuf.clear();
	// Clear metadata
	region = string();
	cluster = string();
	model = string();
	epsgCode = 0;
	origin = glm::vec3(0.0);
	minBB = glm::vec3(FLT_MAX);
	maxBB = glm::vec3(-FLT_MAX);
	atlasSize = glm::uvec2(0);
	satInfo.clear();
	facadeInfo.clear();
}

void Building::genReadMetadata(fs::path inputClusterDir) {
	// Read input metadata file
	fs::path inputMetaPath = inputClusterDir /
		("building_cluster_" + cluster + "__Metadata.json");
	ifstream inputMetaFile(inputMetaPath);
	json metadata;
	inputMetaFile >> metadata;

	// Get EPSG code
	string epsgStr = metadata.at("_items").at("spatial_reference").at("crs").at("data").at("init");
	epsgCode = stoi(epsgStr.substr(5));

	// Get origin
	origin.x = metadata.at("_items").at("spatial_reference").at("affine").at(0);
	origin.y = metadata.at("_items").at("spatial_reference").at("affine").at(3);
	origin.z = metadata.at("_items").at("z_origin");
}

void Building::genGeometry(fs::path inputModelDir, map<string, Satellite>& sats) {
	// Create spatial transformation object
	util::SpatXform sx(epsgCode, origin);

	// Load the obj model
	fs::path objPath = inputModelDir /
		("building_cluster_" + cluster + "__" + model + "__output_mesh.obj");
	tinyobj::attrib_t attrib;
	vector<tinyobj::shape_t> shapes;
	string objWarn, objErr;
	bool objLoaded = tinyobj::LoadObj(&attrib, &shapes, NULL, &objWarn, &objErr,
		objPath.string().c_str(), NULL, true);
	// Check for errors
	if (!objLoaded) {
		stringstream ss;
		ss << "Failed to load " << objPath.filename().string() << ":" << endl;
		ss << objErr;
		throw runtime_error(ss.str());
	}
	// Print any warnings
//	if (!objWarn.empty())
//		cout << objWarn << endl;

	// Get satellite bounding rects
	for (auto& si : sats) {
		Satellite& sat = si.second;

		// Project all vertices
		vector<cv::Point2f> allPts;
		for (size_t v = 0; v < attrib.vertices.size(); v += 3) {
			glm::vec3 pt;
			pt.x = attrib.vertices[v + 0];
			pt.y = attrib.vertices[v + 1];
			pt.z = attrib.vertices[v + 2];

			pt = sx.utm2px(pt, sat);
			allPts.push_back({ pt.x, pt.y });
		}

		// Calculate bounding rect
		cv::Rect bb = sat.calcBB(allPts);
		// Don't use sat if all points are outside ROI
		if (bb.width <= 0 || bb.height <= 0) continue;

		// Save satellite info
		satInfo[sat.name].name = sat.name;
		satInfo[sat.name].roi = bb;
	}

	// Add faces to geometry buffers
	for (size_t s = 0; s < shapes.size(); s++) {
		size_t idx_offset = 0;
		for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
			int fv = shapes[s].mesh.num_face_vertices[f];
			map<string, vector<glm::vec2>> satPts;

			// Skip degenerate and down-facing triangles
			{ tinyobj::index_t ia = shapes[s].mesh.indices[idx_offset + 0];
			glm::vec3 va;
			va.x = attrib.vertices[3 * ia.vertex_index + 0];
			va.y = attrib.vertices[3 * ia.vertex_index + 1];
			va.z = attrib.vertices[3 * ia.vertex_index + 2];
			tinyobj::index_t ib = shapes[s].mesh.indices[idx_offset + 1];
			glm::vec3 vb;
			vb.x = attrib.vertices[3 * ib.vertex_index + 0];
			vb.y = attrib.vertices[3 * ib.vertex_index + 1];
			vb.z = attrib.vertices[3 * ib.vertex_index + 2];
			tinyobj::index_t ic = shapes[s].mesh.indices[idx_offset + 2];
			glm::vec3 vc;
			vc.x = attrib.vertices[3 * ic.vertex_index + 0];
			vc.y = attrib.vertices[3 * ic.vertex_index + 1];
			vc.z = attrib.vertices[3 * ic.vertex_index + 2];

			// Sort edge lengths
			vector<float> lengths;
			lengths.push_back(glm::length(vb - va));
			lengths.push_back(glm::length(vc - vb));
			lengths.push_back(glm::length(va - vc));
			sort(lengths.begin(), lengths.end());
			// Degenerate if largest edge <= sum of smaller edges
			if (lengths[0] + lengths[1] <= lengths[2] + 1e-4) {
				idx_offset += fv;
				cout << "Degenerate triangle! "
					<< lengths[0] << " " << lengths[1] << " " << lengths[2] << endl;
				continue;
			}

			// Skip if facing downward
			glm::vec3 normal = glm::normalize(glm::cross(vb - va, vc - va));
			if (1.0 - glm::dot(normal, { 0.0, 0.0, -1.0 }) < 1e-4) {
				idx_offset += fv;
				continue;
			}}

			// Assume this triangle faces away from all satellite images
			bool allBackfacing = true;

			// Iterate through all saved sats
			for (auto& si : satInfo) {
				string satName = si.first;
				vector<glm::vec2> projPts;

				// Iterate over all verts on this face
				for (size_t v = 0; v < fv; v++) {
					tinyobj::index_t idx = shapes[s].mesh.indices[idx_offset + v];

					// Get the vert coordinates
					glm::vec3 pt;
					pt.x = attrib.vertices[3 * idx.vertex_index + 0];
					pt.y = attrib.vertices[3 * idx.vertex_index + 1];
					pt.z = attrib.vertices[3 * idx.vertex_index + 2];

					// Project onto the satellite image
					pt = sx.utm2uv(pt, sats[satName], si.second.roi);
					projPts.push_back(glm::vec2(pt));
				}

				// Get facing direction
				glm::vec2 e1 = projPts[1] - projPts[0];
				glm::vec2 e2 = projPts[2] - projPts[0];
				// Skip this sat if tri faces backwards
				if (e1.x * e2.y - e1.y * e2.x < 0.0) continue;

				allBackfacing = false;

				// Collect projected points for this sat
				for (auto p : projPts)
					satPts[satName].push_back(p);
			}

			// Skip triangle if no sat saw it
			if (allBackfacing) {
				idx_offset += fv;
				continue;
			}

			// Add face to index buffer in fan configuration
			for (size_t v = 2; v < fv; v++) {
				indexBuf.push_back(posBuf.size());
				indexBuf.push_back(posBuf.size() + v - 1);
				indexBuf.push_back(posBuf.size() + v - 0);
			}
			// Add vertex attributes
			for (size_t v = 0; v < fv; v++) {
				tinyobj::index_t idx = shapes[s].mesh.indices[idx_offset + v];

				// Add position and normal
				posBuf.push_back({
					attrib.vertices[3 * idx.vertex_index + 0],
					attrib.vertices[3 * idx.vertex_index + 1],
					attrib.vertices[3 * idx.vertex_index + 2]});
				normBuf.push_back({
					attrib.normals[3 * idx.normal_index + 0],
					attrib.normals[3 * idx.normal_index + 1],
					attrib.normals[3 * idx.normal_index + 2]});

				// Add all texture coordinates
				for (auto& si : satInfo) {
					string satName = si.first;
					// If face not seen by this sat, pass -1s
					if (!satPts.count(satName))
						satTCBufs[satName].push_back({ -1.0, -1.0 });
					// Otherwise give it the projected coordinates
					else
						satTCBufs[satName].push_back(satPts[satName][v]);
				}

				// Update bounding box
				minBB = glm::min(minBB, posBuf.back());
				maxBB = glm::max(maxBB, posBuf.back());
			}

			idx_offset += fv;
		}
	}

	// Check if result has no geometry
	if (indexBuf.empty()) {
		clear();
		throw runtime_error("No faces!");
	}
}

// Group faces into facades and generate atlas coordinates
void Building::genFacades() {
	// "Fuzzy" plane coefficient comparator
	auto planeCmp = [](const glm::vec4& a, const glm::vec4& b) -> bool {
		static const float eps = 1e-4;
		if (abs(a.x - b.x) > eps) return a.x < b.x;
		if (abs(a.y - b.y) > eps) return a.y < b.y;
		if (abs(a.z - b.z) > eps) return a.z < b.z;
		if (abs(a.w - b.w) > eps) return a.w < b.w;
		return false;
	};
	// Map plane coefficients to groups of triangles
	map<glm::vec4, util::SurfaceGroup, decltype(planeCmp)> groupMap(planeCmp);

	// Group all faces by planes
	for (size_t f = 0; f < indexBuf.size() / 3; f++) {
		// Get the plane coefficients
		glm::vec3 va = posBuf[indexBuf[3 * f + 0]];
		glm::vec3 vb = posBuf[indexBuf[3 * f + 1]];
		glm::vec3 vc = posBuf[indexBuf[3 * f + 2]];
		glm::vec3 norm = glm::normalize(glm::cross(vb - va, vc - va));
		float d = glm::dot(norm, va);
		glm::vec4 plane(norm, d);

		// Add to group
		groupMap[plane].faceIDs.push_back(f);
	}

	// Track group statistics
	util::GroupStats gs;

	// Rectify each group
	for (auto& g : groupMap) {

		// Get rotation
		glm::vec3 norm = glm::normalize(glm::vec3(g.first));
		g.second.xform = glm::mat4(1.0);
		// Don't rotate if the normal is already +Z
		if (glm::dot(norm, { 0.0, 0.0, 1.0 }) < 1.0) {
			// Get orthonormal vectors
			glm::vec3 up(0.0, 0.0, 1.0);
			glm::vec3 right = glm::cross(up, norm);
			up = glm::cross(norm, right);
			// Construct rotation matrix
			g.second.xform[0] = glm::vec4(right, 0.0);
			g.second.xform[1] = glm::vec4(up, 0.0);
			g.second.xform[2] = glm::vec4(norm, 0.0);
			g.second.xform = glm::transpose(g.second.xform);
		}

		// Calculate longest projected edge in all satellite images
		float projEdgeLen = 0.0;
		float rectEdgeLen = 0.0;
		// Iterate over satellites
		for (auto& si : satInfo) {
			// Iterate over faces in this group
			for (auto f : g.second.faceIDs) {
				// Iterate over vertices on this face
				for (size_t vi = 0; vi < 3; vi++) {
					size_t vi2 = (vi + 1) % 3;
					// Get projected length of this edge
					glm::vec2 pa = satTCBufs[si.first][indexBuf[3 * f + vi]];
					glm::vec2 pb = satTCBufs[si.first][indexBuf[3 * f + vi2]];
					pa = glm::vec2(util::SpatXform::uv2px(glm::vec3(pa, 0.0), si.second.roi));
					pb = glm::vec2(util::SpatXform::uv2px(glm::vec3(pb, 0.0), si.second.roi));
					float len = glm::length(pb - pa);
					if (len > projEdgeLen) {
						projEdgeLen = len;

						// Get the length of the same edge after rectifying
						glm::vec3 va = posBuf[indexBuf[3 * f + vi]];
						glm::vec3 vb = posBuf[indexBuf[3 * f + vi2]];
						va = glm::vec3(g.second.xform * glm::vec4(va, 1.0));
						vb = glm::vec3(g.second.xform * glm::vec4(vb, 1.0));
						rectEdgeLen = glm::length(glm::vec2(vb - va));
					}
				}
			}
		}

		// Scale by the ratio of longest lengths
		glm::mat4 scale = glm::mat4(1.0);
		scale[0][0] = projEdgeLen / rectEdgeLen;
		scale[1][1] = projEdgeLen / rectEdgeLen;
		g.second.xform = scale * g.second.xform;

		// Get 2D bounding box
		glm::vec2 minBBf(FLT_MAX), maxBBf(-FLT_MAX);
		for (auto f : g.second.faceIDs) {
			for (size_t vi = 0; vi < 3; vi++) {
				glm::vec4 v = g.second.xform * glm::vec4(posBuf[indexBuf[3 * f + vi]], 1.0);
				minBBf = glm::min(minBBf, glm::vec2(v));
				maxBBf = glm::max(maxBBf, glm::vec2(v));
			}
		}
		// Convert bounding box to pixels, add padding to all sides
		glm::ivec2 padding(2);
		g.second.minBB = glm::ivec2(glm::floor(minBBf)) - padding;
		g.second.maxBB = glm::ivec2(glm::ceil(maxBBf)) + padding;

		// Translate to origin
		glm::mat4 xlate = glm::mat4(1.0);
		xlate[3] = glm::vec4(-g.second.minBB, 0.0f, 1.0f);
		g.second.xform = xlate * g.second.xform;
		// Update bounding box
		g.second.maxBB -= g.second.minBB;
		g.second.minBB = glm::ivec2(0);
		// Update stats
		gs.ta += g.second.maxBB.x * g.second.maxBB.y;
		gs.tw += g.second.maxBB.x;
		gs.th += g.second.maxBB.y;
		gs.mw = glm::max(gs.mw, g.second.maxBB.x);
		gs.mh = glm::max(gs.mh, g.second.maxBB.y);
	}

	// Sort groups by descending height
	vector<util::SurfaceGroup*> groupsByHeight;
	for (auto& g : groupMap) groupsByHeight.push_back(&g.second);
	sort(groupsByHeight.begin(), groupsByHeight.end(),
		[](const util::SurfaceGroup* a, const util::SurfaceGroup* b) -> bool {
			return (a->maxBB.y - a->minBB.y) > (b->maxBB.y - b->minBB.y);
		});

	// Try to find the smallest area we can pack groups onto
	glm::ivec2 trySize(gs.tw, gs.mh);	// Start with shortest, widest possible canvas
	glm::ivec2 bestSize = trySize;
	// Loop until we achieve narrowest canvas
	while (trySize.x >= gs.mw) {
		// Attempt to pack all groups onto the canvas
		util::Canvas canvas(trySize, gs);
		bool packed = canvas.pack(groupsByHeight);

		// We succeeded in packing all groups, try with a narrower canvas
		if (packed) {
			trySize = canvas.size();	// Trim any extra width
			if (trySize.x * trySize.y <= bestSize.x * bestSize.y)
				bestSize = trySize;		// Record the smallet area so far

			// Reduce the width
			trySize.x -= 1;
			// Increase the height by that of the tallest right-most group
			int gi = canvas.idxOfTallestRightGroup();
			if (gi >= 0) {
				int gh = groupsByHeight[gi]->maxBB.y - groupsByHeight[gi]->minBB.y;
				trySize.y += gh;
			}

		// Failed to pack all groups, increase canvas height
		} else {
			int mhd = canvas.minHeightDeficit();
			int gi = canvas.idxOfUnplacedGroup();
			int gih = groupsByHeight[gi]->maxBB.y - groupsByHeight[gi]->minBB.y;
			// Increase height by smaller of min height deficit, or height of unplaced group
			trySize.y += max(min(mhd, gih), 1);
		}

		// Make sure canvas isn't too small or too big
		bool resized = true;
		while (resized) {
			resized = false;
			// Area is smaller than total group area, increase height
			while (trySize.x * trySize.y < gs.ta) {
				trySize.y += 1;
				resized = true;
			}
			// Area is larger than best area so far, reduce width
			while (trySize.x * trySize.y > bestSize.x * bestSize.y) {
				trySize.x -= 1;
				resized = true;
			}
		}
	}

	// Pack groups using the best size we found
	util::Canvas canvas(bestSize, gs);
	bool packed = canvas.pack(groupsByHeight, true);
	canvas.trim(true);
	bestSize = canvas.size();
	atlasSize = bestSize;

	// Scale each group to UV coords
	glm::mat4 px2uv = glm::mat4(1.0);
	px2uv[0][0] = 1.0 / atlasSize.x;
	px2uv[1][1] = 1.0 / atlasSize.y;
	atlasTCBuf.resize(posBuf.size(), glm::vec2(-1.0));
	for (auto& g : groupMap) {
		// Apply px2uv transform
		g.second.xform = px2uv * g.second.xform;

		// Transform each vertex and store in atlas TC buffer
		glm::vec2 uvMinBB(FLT_MAX), uvMaxBB(-FLT_MAX);
		float minH(FLT_MAX), maxH(-FLT_MAX);
		for (auto f : g.second.faceIDs) {
			for (size_t vi = 0; vi < 3; vi++) {
				glm::vec3 v = posBuf[indexBuf[3 * f + vi]];
				glm::vec2 av = glm::vec2(g.second.xform * glm::vec4(v, 1.0));
				atlasTCBuf[indexBuf[3 * f + vi]] = av;

				// Update UV bounding box
				uvMinBB = glm::min(uvMinBB, glm::vec2(av));
				uvMaxBB = glm::max(uvMaxBB, glm::vec2(av));
				// Update height bounds
				minH = glm::min(minH, v.z);
				maxH = glm::max(maxH, v.z);
			}
		}

		// Store facade info
		FacadeInfo fi;
		fi.faceIDs = g.second.faceIDs;
		fi.normal = glm::normalize(glm::vec3(g.first));
		fi.size = glm::uvec2(g.second.maxBB - g.second.minBB);
		fi.atlasBB.x = uvMinBB.x;
		fi.atlasBB.y = uvMinBB.y;
		fi.atlasBB.width = (uvMaxBB - uvMinBB).x;
		fi.atlasBB.height = (uvMaxBB - uvMinBB).y;
		fi.height = maxH - minH;
		fi.ground = (abs(minH - minBB.z) < 1e-4);
		fi.roof = (glm::dot(fi.normal, { 0.0, 0.0, 1.0 }) > 0.707f);	// Roof if < ~45 deg from +Z
		facadeInfo.push_back(fi);
	}
}

// Writes building data to the data directory
void Building::genWriteData(fs::path dataDir) {
	// Create data directories as needed
	modelDir = dataDir / "regions" / region / cluster / model;
	if (!fs::exists(modelDir))
		fs::create_directories(modelDir);

	// Create the output .obj file
	fs::path objPath = modelDir /
		(region + "_" + cluster + "_" + model + ".obj");
	ofstream objFile(objPath);

	// Write all vertex positions
	for (auto v : posBuf)
		objFile << setprecision(20) << "v " << v.x << " " << v.y << " " << v.z << endl;
	objFile << endl;
	// Write all normals
	for (auto n : normBuf)
		objFile << setprecision(20) << "vn " << n.x << " " << n.y << " " << n.z << endl;
	objFile << endl;

	// Keep track of texture coordinate indices
	int tcIdx = 0;

	// Write all sets of texture coordinates
	for (auto& si : satInfo) {

		// Create a group for this satellite image
		objFile << "g " << si.first << endl;

		// Loop through each face
		for (size_t f = 0; f < indexBuf.size(); f += 3) {
			glm::vec2 ta = satTCBufs[si.first][indexBuf[f + 0]];
			glm::vec2 tb = satTCBufs[si.first][indexBuf[f + 1]];
			glm::vec2 tc = satTCBufs[si.first][indexBuf[f + 2]];

			// Skip if no projection
			if (ta.x < 0.0 || ta.y < 0.0 ||
				tb.x < 0.0 || tb.y < 0.0 ||
				tc.x < 0.0 || tc.y < 0.0) continue;

			// Write texture coordinates
			objFile << setprecision(20) << "vt " << ta.x << " " << ta.y << endl;
			objFile << setprecision(20) << "vt " << tb.x << " " << tb.y << endl;
			objFile << setprecision(20) << "vt " << tc.x << " " << tc.y << endl;
			// Write face
			glm::uint idx0 = indexBuf[f + 0];
			glm::uint idx1 = indexBuf[f + 1];
			glm::uint idx2 = indexBuf[f + 2];
			objFile << "f " << idx0 + 1 << "/" << (tcIdx++) + 1 << "/" << idx0 + 1 << " ";
			objFile << idx1 + 1 << "/" << (tcIdx++) + 1 << "/" << idx1 + 1 << " ";
			objFile << idx2 + 1 << "/" << (tcIdx++) + 1 << "/" << idx2 + 1 << endl;
		}
		objFile << endl;
	}

	// Write atlas texture coordinates
	objFile << "g atlas" << endl;
	for (auto t : atlasTCBuf)
		objFile << setprecision(20) << "vt " << t.x << " " << t.y << endl;
	// Write faces
	for (size_t f = 0; f < indexBuf.size(); f += 3) {
		glm::uint idx0 = indexBuf[f + 0];
		glm::uint idx1 = indexBuf[f + 1];
		glm::uint idx2 = indexBuf[f + 2];
		objFile << "f " << idx0 + 1 << "/" << idx0 + tcIdx + 1 << "/" << idx0 + 1 << " ";
		objFile << idx1 + 1 << "/" << idx1 + tcIdx + 1 << "/" << idx1 + 1 << " ";
		objFile << idx2 + 1 << "/" << idx2 + tcIdx + 1 << "/" << idx2 + 1 << endl;
	}
	objFile << endl;


	// Create output metadata
	json meta;
	meta["region"] = region;
	meta["cluster"] = cluster;
	meta["model"] = model;
	meta["epsgCode"] = epsgCode;
	meta["origin"][0] = origin.x;
	meta["origin"][1] = origin.y;
	meta["origin"][2] = origin.z;
	meta["minBB"][0] = minBB.x;
	meta["minBB"][1] = minBB.y;
	meta["minBB"][2] = minBB.z;
	meta["maxBB"][0] = maxBB.x;
	meta["maxBB"][1] = maxBB.y;
	meta["maxBB"][2] = maxBB.z;
	meta["atlasSize"][0] = atlasSize.x;
	meta["atlasSize"][1] = atlasSize.y;

	// Satellite info
	for (auto& si : satInfo) {
		json satInfoMeta;
		satInfoMeta["name"] = si.second.name;
		satInfoMeta["roi"][0] = si.second.roi.x;
		satInfoMeta["roi"][1] = si.second.roi.y;
		satInfoMeta["roi"][2] = si.second.roi.width;
		satInfoMeta["roi"][3] = si.second.roi.height;
		satInfoMeta["dir"][0] = si.second.dir.x;
		satInfoMeta["dir"][1] = si.second.dir.y;
		satInfoMeta["dir"][2] = si.second.dir.z;
		satInfoMeta["sun"][0] = si.second.sun.x;
		satInfoMeta["sun"][1] = si.second.sun.y;
		satInfoMeta["sun"][2] = si.second.sun.z;
		meta["satInfo"].push_back(satInfoMeta);
	}

	// Facade info
	for (auto& fi : facadeInfo) {
		json facadeInfoMeta;
		for (auto f : fi.faceIDs)
			facadeInfoMeta["faceIDs"].push_back(f);
		facadeInfoMeta["normal"][0] = fi.normal.x;
		facadeInfoMeta["normal"][1] = fi.normal.y;
		facadeInfoMeta["normal"][2] = fi.normal.z;
		facadeInfoMeta["size"][0] = fi.size.x;
		facadeInfoMeta["size"][1] = fi.size.y;
		facadeInfoMeta["atlasBB"][0] = fi.atlasBB.x;
		facadeInfoMeta["atlasBB"][1] = fi.atlasBB.y;
		facadeInfoMeta["atlasBB"][2] = fi.atlasBB.width;
		facadeInfoMeta["atlasBB"][3] = fi.atlasBB.height;
		facadeInfoMeta["height"] = fi.height;
		facadeInfoMeta["ground"] = fi.ground;
		facadeInfoMeta["roof"] = fi.roof;
		meta["facadeInfo"].push_back(facadeInfoMeta);
	}

	// Write metadata to file
	fs::path metaPath = objPath;
	metaPath.replace_extension(".json");
	ofstream metaFile(metaPath);
	metaFile << meta << endl;
}

// Read the .obj file and populate the geometry buffers
void Building::loadGeometry(fs::path objPath) {

	// Load the OBJ file
	tinyobj::attrib_t attrib;
	vector<tinyobj::shape_t> shapes;
	string objWarn, objErr;
	bool objLoaded = tinyobj::LoadObj(&attrib, &shapes, NULL, &objWarn, &objErr,
		objPath.string().c_str());
	// Check for errors
	if (!objLoaded) {
		stringstream ss;
		ss << "Failed to load " << objPath.filename().string() << ":" << endl;
		ss << objErr;
		throw runtime_error(ss.str());
	}
	// Print any warnings
//	if (!objWarn.empty())
//		cout << objWarn << endl;

	// Stores indices for all versions of a face
	struct faceIds {
		vector<int> normIds;
		vector<int> atlasTCIds;
		map<string, vector<int>> satTCIds;
	};
	// Maps vertex indices to norm / TC indices
	map<vector<int>, faceIds> faceMap;

	// Loop over all shapes
	for (size_t s = 0; s < shapes.size(); s++) {
		// Touch the sat TC buf for this shape to make sure it exists
		if (shapes[s].name != "atlas")
			satTCBufs[shapes[s].name];

		// Loop over faces
		size_t idx_offset = 0;
		for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
			int fv = shapes[s].mesh.num_face_vertices[f];

			// Store indices for positions, normals, and texture coordinates
			vector<int> posIds;
			vector<int> normIds;
			vector<int> texIds;
			// Loop thru all verts to store the above
			for (size_t v = 0; v < fv; v++) {
				tinyobj::index_t idx = shapes[s].mesh.indices[idx_offset + v];
				posIds.push_back(idx.vertex_index);
				normIds.push_back(idx.normal_index);
				texIds.push_back(idx.texcoord_index);
			}
			idx_offset += fv;

			// Set normal ids if we haven't seen this face before
			if (!faceMap.count(posIds)) {
				faceMap[posIds].normIds = normIds;
			}
			// Set atlas TC ids if this is the atlas group
			if (shapes[s].name == "atlas")
				faceMap[posIds].atlasTCIds = texIds;
			// Otherwise set the TC ids for the satellite given by the shape name
			else
				faceMap[posIds].satTCIds[shapes[s].name] = texIds;
		}
	}

	// Write to geometry buffers
	for (auto& f : faceMap) {
		// Add vertex indices in a fan configuration
		for (size_t v = 2; v < f.first.size(); v++) {
			indexBuf.push_back(posBuf.size());
			indexBuf.push_back(posBuf.size() + v - 1);
			indexBuf.push_back(posBuf.size() + v - 0);
		}

		// Add attributes for each vertex on this face
		for (size_t v = 0; v < f.first.size(); v++) {
			int vIdx = f.first[v];
			int nIdx = f.second.normIds[v];
			// Add position and normal to buffers
			posBuf.push_back({
				attrib.vertices[3 * vIdx + 0],
				attrib.vertices[3 * vIdx + 1],
				attrib.vertices[3 * vIdx + 2]});
			normBuf.push_back({
				attrib.normals[3 * nIdx + 0],
				attrib.normals[3 * nIdx + 1],
				attrib.normals[3 * nIdx + 2]});

			// Add atlas TCs if they exist for this face
			if (!f.second.atlasTCIds.empty()) {
				int aIdx = f.second.atlasTCIds[v];
				atlasTCBuf.push_back({
					attrib.texcoords[2 * aIdx + 0],
					attrib.texcoords[2 * aIdx + 1]});
			// Otherwise fill with -1s
			} else
				atlasTCBuf.push_back({ -1.0, -1.0 });

			// Loop through all satellite buffers
			for (auto& s : satTCBufs) {
				// Add sat TCs if they exist for this face
				if (!f.second.satTCIds[s.first].empty()) {
					int tIdx = f.second.satTCIds[s.first][v];
					satTCBufs[s.first].push_back({
						attrib.texcoords[2 * tIdx + 0],
						attrib.texcoords[2 * tIdx + 1]});
				// Otherwise fill with -1s
				} else
					satTCBufs[s.first].push_back({ -1.0, -1.0 });
			}
		}
	}
}

void Building::loadMetadata(fs::path metaPath) {
	// Read metadata from JSON file
	json meta;
	ifstream metaFile(metaPath);
	metaFile >> meta;

	// Region, cluster, model
	region = meta.at("region");
	cluster = meta.at("cluster");
	model = meta.at("model");
	// EPSG code
	epsgCode = meta.at("epsgCode");
	// Origin in UTM
	origin.x = meta.at("origin").at(0);
	origin.y = meta.at("origin").at(1);
	origin.z = meta.at("origin").at(2);
	// Minimum bounding box
	minBB.x = meta.at("minBB").at(0);
	minBB.y = meta.at("minBB").at(1);
	minBB.z = meta.at("minBB").at(2);
	// Maximum bounding box
	maxBB.x = meta.at("maxBB").at(0);
	maxBB.y = meta.at("maxBB").at(1);
	maxBB.z = meta.at("maxBB").at(2);
	// Size of atlas texture
	atlasSize.x = meta.at("atlasSize").at(0);
	atlasSize.y = meta.at("atlasSize").at(1);

	// Satellite info
	for (size_t s = 0; s < meta.at("satInfo").size(); s++) {
		SatInfo si;

		// Satellite name
		si.name = meta.at("satInfo").at(s).at("name");
		// Region of interest (px, UL origin)
		si.roi.x = meta.at("satInfo").at(s).at("roi").at(0);
		si.roi.y = meta.at("satInfo").at(s).at("roi").at(1);
		si.roi.width = meta.at("satInfo").at(s).at("roi").at(2);
		si.roi.height = meta.at("satInfo").at(s).at("roi").at(3);
		// Satellite facing direction (UTM)
		si.dir.x = meta.at("satInfo").at(s).at("dir").at(0);
		si.dir.y = meta.at("satInfo").at(s).at("dir").at(1);
		si.dir.z = meta.at("satInfo").at(s).at("dir").at(2);
		// Direction towards the sun (UTM)
		si.sun.x = meta.at("satInfo").at(s).at("sun").at(0);
		si.sun.y = meta.at("satInfo").at(s).at("sun").at(1);
		si.sun.z = meta.at("satInfo").at(s).at("sun").at(2);

		// Add to satellite info
		satInfo[si.name] = si;
	}

	// Facade info
	for (size_t f = 0; f < meta.at("facadeInfo").size(); f++) {
		FacadeInfo fi;

		// List of face IDs in this facade
		for (size_t i = 0; i < meta.at("facadeInfo").at(f).at("faceIDs").size(); i++) {
			fi.faceIDs.push_back(meta.at("facadeInfo").at(f).at("faceIDs").at(i));
		}
		// Normalized facing direction (UTM)
		fi.normal.x = meta.at("facadeInfo").at(f).at("normal").at(0);
		fi.normal.y = meta.at("facadeInfo").at(f).at("normal").at(1);
		fi.normal.z = meta.at("facadeInfo").at(f).at("normal").at(2);
		// Width, height of rectified facade (px)
		fi.size.x = meta.at("facadeInfo").at(f).at("size").at(0);
		fi.size.y = meta.at("facadeInfo").at(f).at("size").at(1);
		// Bounding rect of UV coords in atlas (UV, LL origin)
		fi.atlasBB.x = meta.at("facadeInfo").at(f).at("atlasBB").at(0);
		fi.atlasBB.y = meta.at("facadeInfo").at(f).at("atlasBB").at(1);
		fi.atlasBB.width = meta.at("facadeInfo").at(f).at("atlasBB").at(2);
		fi.atlasBB.height = meta.at("facadeInfo").at(f).at("atlasBB").at(3);
		// Height of the facade (UTM)
		fi.height = meta.at("facadeInfo").at(f).at("height");
		// Whether facade touches the ground
		fi.ground = meta.at("facadeInfo").at(f).at("ground");
		// Whether this is a roof facade
		fi.roof = meta.at("facadeInfo").at(f).at("roof");

		// Add to facade info
		facadeInfo.push_back(fi);
	}
}
