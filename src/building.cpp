#include "building.hpp"
#include "json.hpp"
#include "util.hpp"
#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"
using namespace std;
using json = nlohmann::json;
namespace fs = std::experimental::filesystem;

// Generate building data from input directory, and save it to data directory
void Building::generate(fs::path inputDir, fs::path dataDir, vector<Satellite>& sats,
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

	// Create data directories as needed
	modelDir = dataDir / "regions" / region / cluster / model;
	if (!fs::exists(modelDir))
		fs::create_directories(modelDir);

	this->region = region;
	this->cluster = cluster;
	this->model = model;

	// Read metadata
	genReadMetadata(inputClusterDir);

	// Create spatial transformation data
	util::SpatXform sx(epsgCode, origin);
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
	epsgCode = 0;
	origin = glm::vec3(0.0);
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
	if (!objWarn.empty())
		cout << objWarn << endl;

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
		si.roi.z = meta.at("satInfo").at(s).at("roi").at(2);
		si.roi.w = meta.at("satInfo").at(s).at("roi").at(3);
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

		// Facade ID
		fi.id = meta.at("facadeInfo").at(f).at("id");
		// List of face IDs in this facade
		for (size_t i = 0; i < meta.at("facadeInfo").at(f).at("faces").size(); i++) {
			fi.faces.push_back(meta.at("facadeInfo").at(f).at("faces").at(i));
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
		fi.atlasBB.z = meta.at("facadeInfo").at(f).at("atlasBB").at(2);
		fi.atlasBB.w = meta.at("facadeInfo").at(f).at("atlasBB").at(3);
		// Whether this is a roof facade
		fi.roof = meta.at("facadeInfo").at(f).at("roof");

		// Add to facade info
		facadeInfo[fi.id] = fi;
	}
}
