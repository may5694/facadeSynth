#include <glm/gtc/type_ptr.hpp>
#include <rapidjson/document.h>
#include <rapidjson/istreamwrapper.h>
#include <rapidjson/ostreamwrapper.h>
#include <rapidjson/prettywriter.h>
#include "building.hpp"
#include "util.hpp"
#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"
#include "openglcontext.hpp"
#include "gl46.h"
using namespace std;
namespace fs = std::experimental::filesystem;
namespace rj = rapidjson;

// Generate building data from input directory, and save it to data directory
void Building::generate(fs::path inputDir, fs::path satelliteDir, fs::path dataDir,
	map<string, Satellite>& sats, string region, string cluster, string model) {
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

	// Save satellite, facade, and atlas textures
	genTextures(dataDir, satelliteDir, sats);
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

// Compute the best score of each facade and return metadata
map<size_t, fs::path> Building::scoreFacades(fs::path outputDir) {
	// Map facade IDs to output metadata paths
	map<size_t, fs::path> facadeMap;

	// Create output directories
	fs::path outDir = outputDir / region / cluster / model;
	if (!fs::exists(outDir))
		fs::create_directories(outDir);
	fs::path imageDir = outDir / "image";
	if (fs::exists(imageDir))
		fs::remove_all(imageDir);
	fs::create_directory(imageDir);
	fs::path histeqDir = outDir / "histeq";
	if (fs::exists(histeqDir))
		fs::remove_all(histeqDir);
	fs::create_directory(histeqDir);
	fs::path metaDir = outDir / "metadata";
	if (fs::exists(metaDir))
		fs::remove_all(metaDir);
	fs::create_directory(metaDir);

	int clusterInt = stoi(cluster);

	// Loop over all facades
	for (size_t fi = 0; fi < facadeInfo.size(); fi++) {
		const FacadeInfo& finfo = facadeInfo[fi];
		string fiStr; {
			stringstream ss; ss << setw(4) << setfill('0') << fi;
			fiStr = ss.str();
		}

		// Skip roofs
		if (finfo.roof) continue;

		// Get projected area for each satellite
		map<string, float> areas;
		float maxArea = 0.0;
		for (auto& si : finfo.inSats) {
			float area = 0.0;
			// Iterate over each triangle
			for (auto f : finfo.faceIDs) {
				// Get vertices
				glm::vec2 va = satTCBufs[si][indexBuf[3 * f + 0]];
				va = glm::vec2(util::SpatXform::uv2px(glm::vec3(va, 0.0), satInfo[si].roi));
				glm::vec2 vb = satTCBufs[si][indexBuf[3 * f + 1]];
				vb = glm::vec2(util::SpatXform::uv2px(glm::vec3(vb, 0.0), satInfo[si].roi));
				glm::vec2 vc = satTCBufs[si][indexBuf[3 * f + 2]];
				vc = glm::vec2(util::SpatXform::uv2px(glm::vec3(vc, 0.0), satInfo[si].roi));
				// Calc area
				glm::vec2 ba = vb - va;
				glm::vec2 ca = vc - va;
				area += abs(ba.x * ca.y - ba.y * ca.x) / 2.0;
			}
			if (area > maxArea) maxArea = area;
			areas[si] = area;
		}
		// Skip if area is too small
		if (maxArea < 1e-4) continue;

		// Keep track of best scoring facade
		float maxScore = 0.0;
		string maxScoreIdx;
		cv::Mat maxScoreImage;
		cv::Rect maxScoreRect;
		// Get score for each satellite observation of this facade
		for (auto& si : finfo.inSats) {

			// Load RGBA texture
			fs::path bgraPath = modelDir / "facade" / fiStr / (si + "_ps.png");
			cv::Mat bgraImage = cv::imread(bgraPath.string(), CV_LOAD_IMAGE_UNCHANGED);
			if (!bgraImage.data) {
				cout << "Facade " << fi << " texture " << bgraPath.filename() << " missing!" << endl;
				cout << bgraPath << endl;
				continue;
			}
			bgraImage.convertTo(bgraImage, CV_32F, 1.0 / 255.0);

			// Separate into BGR and A
			cv::Mat bgrImage(bgraImage.size(), CV_32FC3), aImage(bgraImage.size(), CV_32FC1);
			cv::mixChannels(vector<cv::Mat>{ bgraImage }, vector<cv::Mat>{ bgrImage, aImage },
				{ 0, 0, 1, 1, 2, 2, 3, 3 });
			cv::Mat aMask = (aImage > 0.5);

			// Find the largest inscribed rectangle
			cv::Rect inRect = util::findLargestRectangle(aMask);
			bgraImage = bgraImage(inRect);
			bgrImage = bgrImage(inRect);
			aImage = aImage(inRect);
			aMask = aMask(inRect);

			// Load cluster mask
			fs::path clusterMaskPath = modelDir / "facade" / fiStr / (si + "_cid.png");
			cv::Mat clusterMask = cv::imread(clusterMaskPath.string(), CV_LOAD_IMAGE_UNCHANGED);
			cv::Mat unOcc;
			if (!clusterMask.data) {
				cout << "Cluster mask " << clusterMaskPath.filename() << " missing!" << endl;
				unOcc = cv::Mat::ones(bgraImage.size(), CV_32FC1);
			} else {
				clusterMask = clusterMask(inRect);
				unOcc = (clusterMask == clusterInt) | (clusterMask == 0);
				unOcc.convertTo(unOcc, CV_32F, 1.0 / 255.0);
			}

			// Convert to HSV space
			cv::Mat hsvImage;
			cv::cvtColor(bgrImage, hsvImage, cv::COLOR_BGR2HSV);
			cv::Mat hImage(hsvImage.size(), CV_32FC1), vImage(hsvImage.size(), CV_32FC1);
			cv::mixChannels(vector<cv::Mat>{ hsvImage }, vector<cv::Mat>{ hImage, vImage },
				{ 0, 0, 2, 1 });

			cv::Size ks(7, 7);
			// Calculate shadows
			cv::Mat hShadow = hImage.clone();
			hShadow.forEach<float>([](float& p, const int* position) -> void {
				p = pow(cos((p / 360.0 - 0.6) * 2.0 * M_PI) * 0.5 + 0.5, 200.0);
			});
			cv::boxFilter(hShadow, hShadow, -1, ks);
			cv::Mat vShadow = vImage.clone();
			vShadow.forEach<float>([](float& p, const int* position) -> void {
				p = pow(max(0.25 - p, 0.0) / 0.25, 0.5);
			});
			cv::boxFilter(vShadow, vShadow, -1, ks);
			cv::Mat inShadow = hShadow.mul(vShadow).mul(aImage);

			// Calculate brightness
			cv::Mat vBright = vImage.clone();
			vBright.forEach<float>([](float& p, const int* position) -> void {
				p = min(p + 0.5, 1.0);
			});
			cv::boxFilter(vBright, vBright, -1, ks);

			// Calculate score
			float w1 = 0.35;		// Shadow
			float w2 = 0.2;			// Brightness
			float w3 = 0.45;		// Area
			cv::Mat score = unOcc.mul(aImage).mul(
				w1 * (1.0 - inShadow) + w2 * vBright + w3 * areas[si] / maxArea);
			float avgScore = cv::mean(score, aMask)[0];
			// Find max score
			if (avgScore > maxScore) {
				maxScore = avgScore;
				maxScoreIdx = si;
				maxScoreImage = bgrImage;
				maxScoreRect = inRect;
			}
		}

		// Save highest scoring facade
		string imageName; {
			stringstream ss;
			ss << cluster << "_" << fixed << setprecision(4) << maxScore
				<< "_" << fiStr << "_" << maxScoreIdx << ".png";
			imageName = ss.str();
		}
		fs::path imagePath = imageDir / imageName;
		maxScoreImage.convertTo(maxScoreImage, CV_8U, 255.0);
		cv::imwrite(imagePath.string(), maxScoreImage);

		// Generate hist-equalized version
		cv::Mat hsvImage;
		cv::cvtColor(maxScoreImage, hsvImage, cv::COLOR_BGR2HSV);
		cv::Mat vHisteq(maxScoreImage.size(), CV_8UC1);
		cv::mixChannels(vector<cv::Mat>{ hsvImage }, vector<cv::Mat>{ vHisteq }, { 2, 0 });
		cv::equalizeHist(vHisteq, vHisteq);
		cv::mixChannels(vector<cv::Mat>{ vHisteq }, vector<cv::Mat>{ hsvImage }, { 0, 2 });
		cv::Mat histeqImage;
		cv::cvtColor(hsvImage, histeqImage, cv::COLOR_HSV2BGR);
		// Save histeq version
		fs::path histeqPath = histeqDir / imageName;
		cv::imwrite(histeqPath.string(), histeqImage);

		// Create JSON metadata
		rj::Document meta;
		meta.SetObject();
		auto& alloc = meta.GetAllocator();
		meta.AddMember("region", rj::StringRef(region.c_str()), alloc);
		meta.AddMember("cluster", rj::StringRef(cluster.c_str()), alloc);
		meta.AddMember("model", rj::StringRef(model.c_str()), alloc);
		meta.AddMember("facade", rj::Value().SetUint(fi), alloc);
		meta.AddMember("satellite", rj::StringRef(maxScoreIdx.c_str()), alloc);
		meta.AddMember("crop", rj::Value().SetArray(), alloc);
		meta["crop"].PushBack(maxScoreRect.x, alloc);
		meta["crop"].PushBack(maxScoreRect.y, alloc);
		meta["crop"].PushBack(maxScoreRect.width, alloc);
		meta["crop"].PushBack(maxScoreRect.height, alloc);
		meta.AddMember("size", rj::Value().SetArray(), alloc);
		meta["size"].PushBack(maxScoreRect.width * finfo.height / finfo.size.y, alloc);
		meta["size"].PushBack(maxScoreRect.height * finfo.height / finfo.size.y, alloc);
		meta.AddMember("ground", rj::Value().SetBool(finfo.ground
			&& (maxScoreRect.y + maxScoreRect.height == finfo.size.y)), alloc);
		meta.AddMember("score", rj::Value().SetFloat(maxScore), alloc);
		meta.AddMember("imagename", rj::Value().SetString(imagePath.string().c_str(), alloc).Move(), alloc);
		// Write to disk
		fs::path metaPath; {
			stringstream ss;
			ss << cluster << "_" << fiStr << ".json";
			metaPath = metaDir / ss.str();
		}
		ofstream metaFile(metaPath);
		rj::OStreamWrapper osw(metaFile);
		rj::PrettyWriter<rj::OStreamWrapper> writer(osw);
		meta.Accept(writer);
		metaFile << endl;

		// Store facade meta path
		facadeMap[fi] = metaPath;
	}

	return facadeMap;
}

void Building::genReadMetadata(fs::path inputClusterDir) {
	// Read input metadata file
	fs::path inputMetaPath = inputClusterDir /
		("building_cluster_" + cluster + "__Metadata.json");
	ifstream inputMetaFile(inputMetaPath);
	rj::IStreamWrapper isw(inputMetaFile);
	rj::Document metadata;
	metadata.ParseStream(isw);

	// Get EPSG code
	string epsgStr = metadata["_items"]["spatial_reference"]["crs"]["data"]["init"].GetString();
	epsgCode = stoi(epsgStr.substr(5));

	// Get origin
	origin.x = metadata["_items"]["spatial_reference"]["affine"][0].GetFloat();
	origin.y = metadata["_items"]["spatial_reference"]["affine"][3].GetFloat();
	origin.z = metadata["_items"]["z_origin"].GetFloat();
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

		// Keep track of which satellites observe this facade
		set<string> inSats;
		// Iterate over satellites
		for (auto si : satInfo) {
			bool observed = false;
			// Iterate over faces in this group
			for (auto f : g.second.faceIDs) {
				glm::vec2 ta = satTCBufs[si.first][indexBuf[3 * f + 0]];
				glm::vec2 tb = satTCBufs[si.first][indexBuf[3 * f + 1]];
				glm::vec2 tc = satTCBufs[si.first][indexBuf[3 * f + 2]];
				if (ta.x >= 0.0 && ta.y >= 0.0 &&
					tb.x >= 0.0 && tb.y >= 0.0 &&
					tc.x >= 0.0 && tc.y >= 0.0) {
					observed = true;
					break;
				}
			}
			if (!observed) continue;
			inSats.insert(si.first);
		}

		// Store facade info
		FacadeInfo fi;
		fi.faceIDs = g.second.faceIDs;
		fi.inSats = std::move(inSats);
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
	rj::Document meta;
	meta.SetObject();
	auto& alloc = meta.GetAllocator();
	meta.AddMember("region", rj::StringRef(region.c_str()), alloc);
	meta.AddMember("cluster", rj::StringRef(cluster.c_str()), alloc);
	meta.AddMember("model", rj::StringRef(model.c_str()), alloc);
	meta.AddMember("epsgCode", rj::Value().SetUint(epsgCode), alloc);
	meta.AddMember("origin", rj::Value().SetArray(), alloc);
	meta["origin"].PushBack(origin.x, alloc);
	meta["origin"].PushBack(origin.y, alloc);
	meta["origin"].PushBack(origin.z, alloc);
	meta.AddMember("minBB", rj::Value().SetArray(), alloc);
	meta["minBB"].PushBack(minBB.x, alloc);
	meta["minBB"].PushBack(minBB.y, alloc);
	meta["minBB"].PushBack(minBB.z, alloc);
	meta.AddMember("maxBB", rj::Value().SetArray(), alloc);
	meta["maxBB"].PushBack(maxBB.x, alloc);
	meta["maxBB"].PushBack(maxBB.y, alloc);
	meta["maxBB"].PushBack(maxBB.z, alloc);
	meta.AddMember("atlasSize", rj::Value().SetArray(), alloc);
	meta["atlasSize"].PushBack(atlasSize.x, alloc);
	meta["atlasSize"].PushBack(atlasSize.y, alloc);

	// Satellite info
	meta.AddMember("satInfo", rj::Value().SetArray(), alloc);
	for (auto& si : satInfo) {
		rj::Value satInfoMeta;
		satInfoMeta.SetObject();

		satInfoMeta.AddMember("name", rj::StringRef(si.second.name.c_str()), alloc);
		satInfoMeta.AddMember("roi", rj::Value().SetArray(), alloc);
		satInfoMeta["roi"].PushBack(si.second.roi.x, alloc);
		satInfoMeta["roi"].PushBack(si.second.roi.y, alloc);
		satInfoMeta["roi"].PushBack(si.second.roi.width, alloc);
		satInfoMeta["roi"].PushBack(si.second.roi.height, alloc);
		satInfoMeta.AddMember("dir", rj::Value().SetArray(), alloc);
		satInfoMeta["dir"].PushBack(si.second.dir.x, alloc);
		satInfoMeta["dir"].PushBack(si.second.dir.y, alloc);
		satInfoMeta["dir"].PushBack(si.second.dir.z, alloc);
		satInfoMeta.AddMember("sun", rj::Value().SetArray(), alloc);
		satInfoMeta["sun"].PushBack(si.second.sun.x, alloc);
		satInfoMeta["sun"].PushBack(si.second.sun.y, alloc);
		satInfoMeta["sun"].PushBack(si.second.sun.z, alloc);

		meta["satInfo"].PushBack(satInfoMeta.Move(), alloc);
	}

	// Facade info
	meta.AddMember("facadeInfo", rj::Value().SetArray(), alloc);
	for (auto& fi : facadeInfo) {
		rj::Value facadeInfoMeta;
		facadeInfoMeta.SetObject();

		facadeInfoMeta.AddMember("faceIDs", rj::Value().SetArray(), alloc);
		for (auto f : fi.faceIDs)
			facadeInfoMeta["faceIDs"].PushBack(f, alloc);
		facadeInfoMeta.AddMember("inSats", rj::Value().SetArray(), alloc);
		for (auto& s : fi.inSats)
			facadeInfoMeta["inSats"].PushBack(rj::StringRef(s.c_str()), alloc);
		facadeInfoMeta.AddMember("normal", rj::Value().SetArray(), alloc);
		facadeInfoMeta["normal"].PushBack(fi.normal.x, alloc);
		facadeInfoMeta["normal"].PushBack(fi.normal.y, alloc);
		facadeInfoMeta["normal"].PushBack(fi.normal.z, alloc);
		facadeInfoMeta.AddMember("size", rj::Value().SetArray(), alloc);
		facadeInfoMeta["size"].PushBack(fi.size.x, alloc);
		facadeInfoMeta["size"].PushBack(fi.size.y, alloc);
		facadeInfoMeta.AddMember("atlasBB", rj::Value().SetArray(), alloc);
		facadeInfoMeta["atlasBB"].PushBack(fi.atlasBB.x, alloc);
		facadeInfoMeta["atlasBB"].PushBack(fi.atlasBB.y, alloc);
		facadeInfoMeta["atlasBB"].PushBack(fi.atlasBB.width, alloc);
		facadeInfoMeta["atlasBB"].PushBack(fi.atlasBB.height, alloc);
		facadeInfoMeta.AddMember("height", rj::Value().SetFloat(fi.height), alloc);
		facadeInfoMeta.AddMember("ground", rj::Value().SetBool(fi.ground), alloc);
		facadeInfoMeta.AddMember("roof", rj::Value().SetBool(fi.roof), alloc);

		meta["facadeInfo"].PushBack(facadeInfoMeta.Move(), alloc);
	}

	// Write metadata to file
	fs::path metaPath = objPath;
	metaPath.replace_extension(".json");
	ofstream metaFile(metaPath);
	rj::OStreamWrapper osw(metaFile);
	rj::PrettyWriter<rj::OStreamWrapper> writer(osw);
	meta.Accept(writer);
	metaFile << endl;
}

// Save cropped versions of all satellite images and masks
void Building::genTextures(fs::path dataDir, fs::path satelliteDir, map<string, Satellite>& sats) {
	// Create directory for satellite images
	fs::path satDir = modelDir / "sat";
	if (!fs::exists(satDir))
		fs::create_directory(satDir);

	// Iterate over all used satellites
	for (auto& si : satInfo) {
		Satellite& sat = sats.at(si.first);

		// Save a cropped version of the image
		fs::path satPath = satDir / (si.second.name + "_ps.png");
		cv::imwrite(satPath.string(), sat.satImg(si.second.roi));
	}

	// Look for cluster masks
	fs::path clusterMasksDir = satelliteDir / region / "clusterMasks" / model;
	set<string> masksFound;
	fs::directory_iterator di(clusterMasksDir), dend;
	for (; di != dend; ++di) {
		// Skip if not an image file with a matching prefix to a sat dataset
		if (!fs::is_regular_file(di->path())) continue;
		if (di->path().extension().string() != ".png") continue;
		string prefix = di->path().filename().string().substr(0, 13);
		if (!satInfo.count(prefix)) continue;

		// Save a cropped version of the cluster mask
		fs::path maskPath = satDir / (prefix + "_cid.png");
		cv::Mat cmask = cv::imread(di->path().string(), CV_LOAD_IMAGE_UNCHANGED);
		cv::imwrite(maskPath.string(), cmask(satInfo[prefix].roi));
		masksFound.insert(prefix);
	}
	// Warn for all masks not found
	for (auto& si : satInfo)
		if (!masksFound.count(si.first))
			cout << "Cluster mask for " << si.first << " missing!" << endl;


	// Initialize OpenGL
	OpenGLContext ctx;
	glClearColor(0.0, 0.0, 0.0, 0.0);
	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
	glPixelStorei(GL_PACK_ALIGNMENT, 1);

	// Create framebuffer texture
	GLuint fbtex = ctx.genTexture();
	glBindTexture(GL_TEXTURE_2D, fbtex);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 1, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glBindTexture(GL_TEXTURE_2D, 0);
	// Create framebuffer object
	GLuint fbo = ctx.genFramebuffer();
	glBindFramebuffer(GL_FRAMEBUFFER, fbo);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, fbtex, 0);

	// Create vertex buffers
	GLuint atbuf = ctx.genBuffer();
	glBindBuffer(GL_ARRAY_BUFFER, atbuf);
	glBufferData(GL_ARRAY_BUFFER, atlasTCBuf.size() * sizeof(atlasTCBuf[0]),
		atlasTCBuf.data(), GL_STATIC_DRAW);
	map<string, GLuint> stbufs;
	for (auto& si : satInfo) {
		stbufs[si.first] = ctx.genBuffer();
		glBindBuffer(GL_ARRAY_BUFFER, stbufs[si.first]);
		glBufferData(GL_ARRAY_BUFFER, satTCBufs[si.first].size() * sizeof(satTCBufs[si.first][0]),
			satTCBufs[si.first].data(), GL_STATIC_DRAW);
	}
	GLuint ibuf = ctx.genBuffer();
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibuf);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, indexBuf.size() * sizeof(indexBuf[0]),
		indexBuf.data(), GL_STATIC_DRAW);

	// Create vertex array objects
	map<string, GLuint> svaos;
	for (auto& si : satInfo) {
		svaos[si.first] = ctx.genVAO();
		glBindVertexArray(svaos[si.first]);
		glBindBuffer(GL_ARRAY_BUFFER, atbuf);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(glm::vec2), 0);
		glBindBuffer(GL_ARRAY_BUFFER, stbufs[si.first]);
		glEnableVertexAttribArray(1);
		glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(glm::vec2), 0);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibuf);
	}
	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

	// Vertex shader
	static const string vsh_str = R"(
		#version 460

		layout (location = 0) in vec2 pos;
		layout (location = 1) in vec2 tc;

		smooth out vec2 geoTC;

		uniform mat4 xform;

		void main() {
			gl_Position = xform * vec4(pos, 0.0, 1.0);
			geoTC = tc;
		})";
	// Geometry shader
	static const string gsh_str = R"(
		#version 460

		layout (triangles) in;
		layout (triangle_strip, max_vertices = 3) out;

		smooth in vec2 geoTC[];

		smooth out vec2 fragTC;

		void main() {
			// Skip triangles without valid texture coords
			if (geoTC[0].x < 0.0 || geoTC[0].y < 0.0 ||
				geoTC[1].x < 0.0 || geoTC[1].y < 0.0 ||
				geoTC[2].x < 0.0 || geoTC[2].y < 0.0) return;

			gl_Position = gl_in[0].gl_Position;
			fragTC = geoTC[0];
			EmitVertex();

			gl_Position = gl_in[1].gl_Position;
			fragTC = geoTC[1];
			EmitVertex();

			gl_Position = gl_in[2].gl_Position;
			fragTC = geoTC[2];
			EmitVertex();

			EndPrimitive();
		})";
	// Fragment shader
	static const string fsh_str = R"(
		#version 460

		smooth in vec2 fragTC;

		out vec4 outCol;

		uniform sampler2D texSampler;

		void main() {
			outCol = texture(texSampler, fragTC);
		})";

	// Compile and link shaders
	vector<GLuint> shaders;
	shaders.push_back(ctx.compileShader(GL_VERTEX_SHADER, vsh_str));
	shaders.push_back(ctx.compileShader(GL_GEOMETRY_SHADER, gsh_str));
	shaders.push_back(ctx.compileShader(GL_FRAGMENT_SHADER, fsh_str));
	GLuint program = ctx.linkProgram(shaders);
	GLuint xformLoc = glGetUniformLocation(program, "xform");
	glUseProgram(program);

	// Upload satellite images and masks to textures
	map<string, GLuint> satTexs;
	map<string, GLuint> maskTexs;
	for (auto& si : satInfo) {
		fs::path satPath = satDir / (si.first + "_ps.png");
		cv::Mat satImg = cv::imread(satPath.string(), CV_LOAD_IMAGE_UNCHANGED);
		cv::flip(satImg, satImg, 0);

		satTexs[si.first] = ctx.genTexture();
		glBindTexture(GL_TEXTURE_2D, satTexs[si.first]);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, satImg.cols, satImg.rows, 0,
			GL_BGR, GL_UNSIGNED_BYTE, satImg.data);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

		fs::path maskPath = satDir / (si.first + "_cid.png");
		cv::Mat maskImg = cv::imread(maskPath.string(), CV_LOAD_IMAGE_UNCHANGED);
		if (!maskImg.data) continue;
		cv::flip(maskImg, maskImg, 0);

		maskTexs[si.first] = ctx.genTexture();
		glBindTexture(GL_TEXTURE_2D, maskTexs[si.first]);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, maskImg.cols, maskImg.rows, 0,
			GL_RED, GL_UNSIGNED_BYTE, maskImg.data);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	}


	// Create facade directory
	fs::path facadeDir = modelDir / "facade";
	if (!fs::exists(facadeDir))
		fs::create_directory(facadeDir);
	// Iterate over all facades
	for (size_t fi = 0; fi < facadeInfo.size(); fi++) {
		// Create facade ID directory
		fs::path facadeIDDir; {
			stringstream ss; ss << setw(4) << setfill('0') << fi;
			facadeIDDir = facadeDir / ss.str();
		}
		if (!fs::exists(facadeIDDir))
			fs::create_directory(facadeIDDir);

		// Resize framebuffer texture
		glBindTexture(GL_TEXTURE_2D, fbtex);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, facadeInfo[fi].size.x, facadeInfo[fi].size.y, 0,
			GL_RGBA, GL_UNSIGNED_BYTE, NULL);
		glBindTexture(GL_TEXTURE_2D, 0);
		// Set viewport
		glViewport(0, 0, facadeInfo[fi].size.x, facadeInfo[fi].size.y);

		// Setup transformation matrix
		cv::Rect2f atlasBB = facadeInfo[fi].atlasBB;
		glm::mat4 xlate(1.0);
		xlate[3] = glm::vec4(
			-(atlasBB.x + atlasBB.width / 2.0),
			-(atlasBB.y + atlasBB.height / 2.0),
			0.0, 1.0);
		glm::mat4 scale(1.0);
		scale[0][0] = 2.0 / atlasBB.width;
		scale[1][1] = 2.0 / atlasBB.height;
		glm::mat4 xform = scale * xlate;
		glUniformMatrix4fv(xformLoc, 1, GL_FALSE, glm::value_ptr(xform));

		// Iterate over all satellites
		for (auto& si : satInfo) {
			// Skip if not observed by this sat
			if (!facadeInfo[fi].inSats.count(si.first))
				continue;

			// Bind vertex arrays
			glBindVertexArray(svaos[si.first]);
			// Bind satellite texture
			glBindTexture(GL_TEXTURE_2D, satTexs[si.first]);

			glClear(GL_COLOR_BUFFER_BIT);
			// Draw each face in this facade group
			for (auto f : facadeInfo[fi].faceIDs)
				glDrawElements(GL_TRIANGLES, 3, GL_UNSIGNED_INT, (GLvoid*)(3 * f * sizeof(glm::uint)));

			// Download rendered texture
			cv::Mat facadeImg(facadeInfo[fi].size.y, facadeInfo[fi].size.x, CV_8UC4);
			glBindTexture(GL_TEXTURE_2D, fbtex);
			glGetTexImage(GL_TEXTURE_2D, 0, GL_BGRA, GL_UNSIGNED_BYTE, facadeImg.data);
			cv::flip(facadeImg, facadeImg, 0);

			// Save to disk
			fs::path facadePath = facadeIDDir / (si.first + "_ps.png");
			cv::imwrite(facadePath.string(), facadeImg);


			// Skip cluster mask if it doesn't exist
			if (!maskTexs.count(si.first)) continue;

			// Bind cluster mask
			glBindTexture(GL_TEXTURE_2D, maskTexs[si.first]);

			glClear(GL_COLOR_BUFFER_BIT);
			// Draw each face in this facade group
			for (auto f : facadeInfo[fi].faceIDs)
				glDrawElements(GL_TRIANGLES, 3, GL_UNSIGNED_INT, (GLvoid*)(3 * f * sizeof(glm::uint)));

			// Download rendered texture
			cv::Mat maskImg(facadeInfo[fi].size.y, facadeInfo[fi].size.x, CV_8UC1);
			glBindTexture(GL_TEXTURE_2D, fbtex);
			glGetTexImage(GL_TEXTURE_2D, 0, GL_RED, GL_UNSIGNED_BYTE, maskImg.data);
			cv::flip(maskImg, maskImg, 0);

			// Save to disk
			fs::path maskPath = facadeIDDir / (si.first + "_cid.png");
			cv::imwrite(maskPath.string(), maskImg);
		}
	}


	// Resize framebuffer and viewport for atlas textures
	glBindTexture(GL_TEXTURE_2D, fbtex);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, atlasSize.x, atlasSize.y, 0,
		GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	glBindTexture(GL_TEXTURE_2D, 0);
	glViewport(0, 0, atlasSize.x, atlasSize.y);

	// Translate midpoint to 0, 0
	glm::mat4 xlate(1.0);
	xlate[3] = glm::vec4(-0.5, -0.5, 0.0, 1.0);
	// Scale to -1, 1
	glm::mat4 scale(1.0);
	scale[0][0] = 2.0;
	scale[1][1] = 2.0;
	// Combine and set transformation matrix
	glm::mat4 xform = scale * xlate;
	glUniformMatrix4fv(xformLoc, 1, GL_FALSE, glm::value_ptr(xform));

	// Create atlas directory
	fs::path atlasDir = modelDir / "atlas";
	if (!fs::exists(atlasDir))
		fs::create_directory(atlasDir);
	// Iterate over all satellites
	for (auto& si : satInfo) {
		// Bind vertex arrays
		glBindVertexArray(svaos[si.first]);
		// Bind satellite texture
		glBindTexture(GL_TEXTURE_2D, satTexs[si.first]);

		glClear(GL_COLOR_BUFFER_BIT);
		// Draw all triangles
		glDrawElements(GL_TRIANGLES, indexBuf.size(), GL_UNSIGNED_INT, 0);

		// Download rendered texture
		cv::Mat atlasImg(atlasSize.y, atlasSize.x, CV_8UC4);
		glBindTexture(GL_TEXTURE_2D, fbtex);
		glGetTexImage(GL_TEXTURE_2D, 0, GL_BGRA, GL_UNSIGNED_BYTE, atlasImg.data);
		cv::flip(atlasImg, atlasImg, 0);

		// Save to disk
		fs::path atlasPath = atlasDir / (si.first + "_ps.png");
		cv::imwrite(atlasPath.string(), atlasImg);


		// Skip cluster mask if it doesn't exist
		if (!maskTexs.count(si.first)) continue;

		// Bind cluster mask texture
		glBindTexture(GL_TEXTURE_2D, maskTexs[si.first]);

		glClear(GL_COLOR_BUFFER_BIT);
		// Draw all triangles
		glDrawElements(GL_TRIANGLES, indexBuf.size(), GL_UNSIGNED_INT, 0);

		// Download rendered texture
		cv::Mat maskImg(atlasSize.y, atlasSize.x, CV_8UC1);
		glBindTexture(GL_TEXTURE_2D, fbtex);
		glGetTexImage(GL_TEXTURE_2D, 0, GL_RED, GL_UNSIGNED_BYTE, maskImg.data);
		cv::flip(maskImg, maskImg, 0);

		// Save to disk
		fs::path maskPath = atlasDir / (si.first + "_cid.png");
		cv::imwrite(maskPath.string(), maskImg);
	}
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
	ifstream metaFile(metaPath);
	rj::IStreamWrapper isw(metaFile);
	rj::Document meta;
	meta.ParseStream(isw);

	// Region, cluster, model
	region = meta["region"].GetString();
	cluster = meta["cluster"].GetString();
	model = meta["model"].GetString();
	// EPSG code
	epsgCode = meta["epsgCode"].GetUint();
	// Origin in UTM
	origin.x = meta["origin"][0].GetFloat();
	origin.y = meta["origin"][1].GetFloat();
	origin.z = meta["origin"][2].GetFloat();
	// Minimum bounding box
	minBB.x = meta["minBB"][0].GetFloat();
	minBB.y = meta["minBB"][1].GetFloat();
	minBB.z = meta["minBB"][2].GetFloat();
	// Maximum bounding box
	maxBB.x = meta["maxBB"][0].GetFloat();
	maxBB.y = meta["maxBB"][1].GetFloat();
	maxBB.z = meta["maxBB"][2].GetFloat();
	// Size of atlas texture
	atlasSize.x = meta["atlasSize"][0].GetUint();
	atlasSize.y = meta["atlasSize"][1].GetUint();

	// Satellite info
	for (rj::SizeType s = 0; s < meta["satInfo"].Size(); s++) {
		SatInfo si;

		// Satellite name
		si.name = meta["satInfo"][s]["name"].GetString();
		// Region of interest (px, UL origin)
		si.roi.x = meta["satInfo"][s]["roi"][0].GetInt();
		si.roi.y = meta["satInfo"][s]["roi"][1].GetInt();
		si.roi.width = meta["satInfo"][s]["roi"][2].GetInt();
		si.roi.height = meta["satInfo"][s]["roi"][3].GetInt();
		// Satellite facing direction (UTM)
		si.dir.x = meta["satInfo"][s]["dir"][0].GetFloat();
		si.dir.y = meta["satInfo"][s]["dir"][1].GetFloat();
		si.dir.z = meta["satInfo"][s]["dir"][2].GetFloat();
		// Direction towards the sun (UTM)
		si.sun.x = meta["satInfo"][s]["sun"][0].GetFloat();
		si.sun.y = meta["satInfo"][s]["sun"][1].GetFloat();
		si.sun.z = meta["satInfo"][s]["sun"][2].GetFloat();

		// Add to satellite info
		satInfo[si.name] = si;
	}

	// Facade info
	for (size_t f = 0; f < meta["facadeInfo"].Size(); f++) {
		FacadeInfo fi;

		// List of face IDs in this facade
		for (size_t i = 0; i < meta["facadeInfo"][f]["faceIDs"].Size(); i++) {
			fi.faceIDs.push_back(meta["facadeInfo"][f]["faceIDs"][i].GetUint());
		}
		// List of satellites observing this facade
		for (size_t i = 0; i < meta["facadeInfo"][f]["inSats"].Size(); i++) {
			fi.inSats.insert(meta["facadeInfo"][f]["inSats"][i].GetString());
		}
		// Normalized facing direction (UTM)
		fi.normal.x = meta["facadeInfo"][f]["normal"][0].GetFloat();
		fi.normal.y = meta["facadeInfo"][f]["normal"][1].GetFloat();
		fi.normal.z = meta["facadeInfo"][f]["normal"][2].GetFloat();
		// Width, height of rectified facade (px)
		fi.size.x = meta["facadeInfo"][f]["size"][0].GetInt();
		fi.size.y = meta["facadeInfo"][f]["size"][1].GetInt();
		// Bounding rect of UV coords in atlas (UV, LL origin)
		fi.atlasBB.x = meta["facadeInfo"][f]["atlasBB"][0].GetFloat();
		fi.atlasBB.y = meta["facadeInfo"][f]["atlasBB"][1].GetFloat();
		fi.atlasBB.width = meta["facadeInfo"][f]["atlasBB"][2].GetFloat();
		fi.atlasBB.height = meta["facadeInfo"][f]["atlasBB"][3].GetFloat();
		// Height of the facade (UTM)
		fi.height = meta["facadeInfo"][f]["height"].GetFloat();
		// Whether facade touches the ground
		fi.ground = meta["facadeInfo"][f]["ground"].GetBool();
		// Whether this is a roof facade
		fi.roof = meta["facadeInfo"][f]["roof"].GetBool();

		// Add to facade info
		facadeInfo.push_back(fi);
	}
}
