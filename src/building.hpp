#ifndef BUILDING_HPP
#define BUILDING_HPP

#include <vector>
#include <map>
#include <experimental/filesystem>
#include <glm/glm.hpp>
#include <opencv2/opencv.hpp>
#include "satellite.hpp"
namespace fs = std::experimental::filesystem;

struct SatInfo;
struct FacadeInfo;

class Building {
public:
	// Generate building data
	void generate(fs::path clusterDir, fs::path dataDir, std::map<std::string, Satellite>& sats,
		std::string region, std::string cluster, std::string model);
	// Load existing building data
	void load(fs::path dataDir, std::string region, std::string cluster, std::string model);
	// Return building to empty state
	void clear();

	// Score all facades and save metadata for each
	std::map<size_t, fs::path> scoreFacades(fs::path outputDir) const;
	void synthFacadeGeometry(fs::path outputDir, std::map<size_t, fs::path> facades);
	static void combineOutput(fs::path outputDir, std::string region, std::string model,
		std::vector<Building>& bldgs);
	static void createClusterMasks(fs::path dataDir, std::map<std::string, Satellite>& sats,
		std::string region, std::string model, std::vector<Building>& bldgs);

	// Geometry accessors
	const auto& getPosBuf() const { return posBuf; }
	const auto& getNormBuf() const { return normBuf; }
	const auto& getAtlasTCBuf() const { return atlasTCBuf; }
	const auto& getSatTCBufs() const { return satTCBufs; }
	const auto& getIndexBuf() const { return indexBuf; }
	// Metadata accessors
	std::string getRegion() const { return region; }
	std::string getCluster() const { return cluster; }
	std::string getModel() const { return model; }
	uint32_t getEPSGCode() const { return epsgCode; }
	glm::vec3 getOrigin() const { return origin; }
	glm::uvec2 getAtlasSize() const { return atlasSize; }
	const auto& getSatInfo() const { return satInfo; }
	const auto& getFacadeInfo() const { return facadeInfo; }

private:
	fs::path modelDir;		// Directory containing all building data; not saved

	// Geometry buffers
	std::vector<glm::vec3> posBuf;								// Positions
	std::vector<glm::vec3> normBuf;								// Normals
	std::vector<glm::vec2> atlasTCBuf;							// Atlas texture coords
	std::map<std::string, std::vector<glm::vec2>> satTCBufs;	// Satellite texture coords
	std::vector<glm::uint> indexBuf;							// Index buffer

	// Metadata
	std::string region;					// Name of region
	std::string cluster;				// Cluster ID
	std::string model;					// Model name
	uint32_t epsgCode;					// Defines UTM coordinate space
	glm::vec3 origin;					// Geometry origin (UTM)
	glm::vec3 minBB;					// Minimal coordinate in bounding box (UTM)
	glm::vec3 maxBB;					// Maximal coordinate in bounding box (UTM)
	glm::uvec2 atlasSize;				// Width, height of atlas texture (px)
	std::map<std::string, SatInfo> satInfo;		// Per-satellite info
	std::vector<FacadeInfo> facadeInfo;			// Per-facade info

	// Generation methods
	void genReadMetadata(fs::path inputClusterDir);
	void genGeometry(fs::path inputModelDir, std::map<std::string, Satellite>& sats);
	void genFacades();
	void genWriteData(fs::path dataDir);
	void genTextures(fs::path dataDir, std::map<std::string, Satellite>& sats);
	void genClusterMasks(fs::path dataDir);
	// Loading methods
	void loadGeometry(fs::path objPath);
	void loadMetadata(fs::path metaPath);
};

// Holds information about a satellite dataset
struct SatInfo {
	std::string name;		// 13-digit satellite prefix (e.g., "16FEB29012345")
	cv::Rect roi;			// Cropped rect surrouding building (px, UL origin)
	glm::vec3 dir;			// Estimated facing direction (UTM)
	glm::vec3 sun;			// Direction towards the sun (UTM)
};

// Holds information about a facade
struct FacadeInfo {
	std::vector<int> faceIDs;		// List of face IDs within this facade
	std::set<std::string> inSats;	// Which satellites this facade appears in
	glm::vec3 normal;				// Normalized facing direction (UTM)
	glm::uvec2 size;				// Width, height of rectified facade (px)
	cv::Rect2f atlasBB;				// Bounding rect in atlas (UV, LL origin)
	float height;					// Height of facade (UTM)
	bool ground;					// Whether facade touches the ground
	bool roof;						// Whether facade is a roof
};

#endif
