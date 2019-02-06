#ifndef BUILDING_HPP
#define BUILDING_HPP

#include <vector>
#include <map>
#include <experimental/filesystem>
#include <glm/glm.hpp>
#include "satellite.hpp"
namespace fs = std::experimental::filesystem;

struct SatInfo;
struct FacadeInfo;

class Building {
public:
	// Generate building data
	void generate(fs::path inputClusterDir, fs::path dataClusterDir,
		std::string model, std::vector<Satellite>& sats);
	// Load existing building data
	void load(fs::path dataClusterDir, std::string model);

	// Geometry accessors
	const auto& getPosBuf() const { return posBuf; }
	const auto& getNormBuf() const { return normBuf; }
	const auto& getAtlasTCBuf() const { return atlasTCBuf; }
	const auto& getSatTCBufs() const { return satTCBufs; }
	const auto& getIndexBuf() const { return indexBuf; }
	// Metadata accessors
	uint32_t getEPSGCode() const { return epsgCode; }
	glm::vec3 getOrigin() const { return origin; }
	glm::uvec2 getAtlasSize() const { return atlasSize; }
	const auto& getSatInfo() const { return satInfo; }
	const auto& getFacadeInfo() const { return facadeInfo; }

private:
	// Geometry buffers
	std::vector<glm::vec3> posBuf;								// Positions
	std::vector<glm::vec3> normBuf;								// Normals
	std::vector<glm::vec2> atlasTCBuf;							// Atlas texture coords
	std::map<std::string, std::vector<glm::vec2>> satTCBufs;	// Satellite texture coords
	std::vector<glm::uint> indexBuf;							// Index buffer

	// Metadata
	uint32_t epsgCode;							// Defines UTM coordinate space
	glm::vec3 origin;							// Geometry origin (UTM)
	glm::uvec2 atlasSize;						// Width, height of atlas texture (px)
	std::map<std::string, SatInfo> satInfo;		// Per-satellite info
	std::map<uint32_t, FacadeInfo> facadeInfo;	// Per-facade info
};

// Holds information about a satellite dataset
struct SatInfo {
	std::string name;		// 13-digit satellite prefix (e.g., "16FEB29012345")
	glm::uvec4 roi;			// Cropped rect surrouding building (px, UL origin)
	glm::vec3 dir;			// Estimated facing direction (UTM)
	glm::vec3 sun;			// Direction towards the sun (UTM)
};

// Holds information about a facade
struct FacadeInfo {
	uint32_t id;					// Facade ID
	std::vector<glm::uint> faces;	// List of face IDs within this facade
	glm::vec3 normal;				// Normalized facing direction (UTM)
	glm::uvec2 size;				// Width, height of rectified facade (px)
	glm::vec4 atlasBB;				// Bounding rect (x, y, w, h) in atlas (UV, LL origin)
	bool roof;						// Whether facade is a roof
};

#endif
