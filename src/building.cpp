#include <functional>
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
void Building::generate(fs::path clusterDir, fs::path dataDir, map<string, Satellite>& sats,
	string region, string cluster, string model) {
	// Clear any existing contents
	clear();

	// Check for satellite images
	if (sats.empty())
		throw runtime_error("No satellite images!");

	// Check for input directory existence
	fs::path inputClusterDir = clusterDir / cluster;
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
	genTextures(dataDir, sats);

	// Save cropped and facade cluster masks
	genClusterMasks(dataDir);
}

// Load building data from data directory
void Building::load(fs::path dataDir, string region, string cluster, string model) {
	// Clear any existing contents
	clear();

	// Check path to building data
	modelDir = dataDir / "regions" / region / model / cluster;
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
map<size_t, fs::path> Building::scoreFacades(fs::path outputDir) const {
	// Map facade IDs to output metadata paths
	map<size_t, fs::path> facadeMap;

	bool debug = false;

	// Create output directories
	fs::path outDir = outputDir / region / model / cluster;
	if (!fs::exists(outDir))
		fs::create_directories(outDir);
	fs::path imageDir = outDir / "image";
	if (fs::exists(imageDir))
		fs::remove_all(imageDir);
	fs::create_directory(imageDir);
	fs::path imageHisteqDir = outDir / "image-histeq";
	if (fs::exists(imageHisteqDir))
		fs::remove_all(imageHisteqDir);
	fs::create_directory(imageHisteqDir);
	fs::path chipDir = outDir / "chip";
	if (fs::exists(chipDir))
		fs::remove_all(chipDir);
	fs::create_directory(chipDir);
	fs::path chipHisteqDir = outDir / "chip-histeq";
	if (fs::exists(chipHisteqDir))
		fs::remove_all(chipHisteqDir);
	fs::create_directory(chipHisteqDir);
	fs::path metaDir = outDir / "metadata";
	if (fs::exists(metaDir))
		fs::remove_all(metaDir);
	fs::create_directory(metaDir);

	fs::path debugDir = outDir / "debug";
	if (debug) {
		if (fs::exists(debugDir))
			fs::remove_all(debugDir);
		fs::create_directory(debugDir);
	}

	int clusterInt = stoi(cluster);

	// Loop over all facades
	for (size_t fi = 0; fi < facadeInfo.size(); fi++) {
		const FacadeInfo& finfo = facadeInfo[fi];
		string fiStr; {
			stringstream ss; ss << setw(4) << setfill('0') << fi;
			fiStr = ss.str();
		}

		// Get projected area for each satellite
		map<string, float> areas;
		float maxArea = 0.0;
		for (auto& si : finfo.inSats) {
			float area = 0.0;
			// Iterate over each triangle
			for (auto f : finfo.faceIDs) {
				// Get vertices
				glm::vec2 va = satTCBufs.at(si)[indexBuf[3 * f + 0]];
				va = glm::vec2(util::SpatXform::uv2px(glm::vec3(va, 0.0), satInfo.at(si).roi));
				glm::vec2 vb = satTCBufs.at(si)[indexBuf[3 * f + 1]];
				vb = glm::vec2(util::SpatXform::uv2px(glm::vec3(vb, 0.0), satInfo.at(si).roi));
				glm::vec2 vc = satTCBufs.at(si)[indexBuf[3 * f + 2]];
				vc = glm::vec2(util::SpatXform::uv2px(glm::vec3(vc, 0.0), satInfo.at(si).roi));
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
		float maxScore = -1.0;
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
			// Skip empty rectangles
			if (inRect.width <= 0 || inRect.height <= 0)
				continue;
			bgraImage = bgraImage(inRect);
			bgrImage = bgrImage(inRect);
			aImage = aImage(inRect);
			aMask = aMask(inRect);

			// Load cluster mask
			fs::path clusterMaskPath = modelDir / "facade" / fiStr / (si + "_cid.png");
			cv::Mat clusterMask = cv::imread(clusterMaskPath.string(), CV_LOAD_IMAGE_UNCHANGED);
			cv::Mat unOcc;
			if (!clusterMask.data) {
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
				p = min(p * 2.0, 1.0);
			});
			cv::boxFilter(vBright, vBright, -1, ks);

			// Calculate score
			float w1 = 0.35;		// Shadow
			float w2 = 0.35;		// Brightness
			float w3 = 0.30;		// Area
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

			if (debug) {
				// Save every facade with its score
				fs::path debugPath;
				{ stringstream ss;
					ss << cluster << "_" << fiStr << "_"
						<< fixed << setprecision(4) << avgScore << ".png";
				debugPath = debugDir / ss.str(); }
				cv::Mat debugImg;
				bgrImage.convertTo(debugImg, CV_8U, 255.0);
				cv::imwrite(debugPath.string(), debugImg);

				{ stringstream ss;
					ss << cluster << "_" << fiStr << "_s_"
						<< fixed << setprecision(4) << avgScore << ".png";
				debugPath = debugDir / ss.str(); }
				inShadow.convertTo(debugImg, CV_8U, 255.0);
				cv::imwrite(debugPath.string(), debugImg);

				{ stringstream ss;
					ss << cluster << "_" << fiStr << "_b_"
						<< fixed << setprecision(4) << avgScore << ".png";
				debugPath = debugDir / ss.str(); }
				vBright.convertTo(debugImg, CV_8U, 255.0);
				cv::imwrite(debugPath.string(), debugImg);
			}
		}

		// Save highest scoring facade
		string imageName; {
			stringstream ss;
			ss << cluster << "_" << fixed << setprecision(4) << maxScore
				<< "_" << fiStr << "_" << maxScoreIdx << ".png";
			imageName = ss.str();
		}
		fs::path chipPath = chipDir / imageName;
		maxScoreImage.convertTo(maxScoreImage, CV_8U, 255.0);
		cv::imwrite(chipPath.string(), maxScoreImage);

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
		fs::path histeqPath = chipHisteqDir / imageName;
		cv::imwrite(histeqPath.string(), histeqImage);

		// Copy original to output image dir
		fs::path bestOrigPath = modelDir / "facade" / fiStr / (maxScoreIdx + "_ps.png");
		fs::path bestCopyPath = imageDir / (cluster + "_" + fiStr + ".png");
		fs::copy_file(bestOrigPath, bestCopyPath);

		// Create histeq version
		cv::Mat bestBGRAImage = cv::imread(bestCopyPath.string(), CV_LOAD_IMAGE_UNCHANGED);
		cv::Mat bestBGRImage(bestBGRAImage.size(), CV_8UC3);
		cv::Mat bestAImage(bestBGRAImage.size(), CV_8UC1);
		cv::mixChannels(vector<cv::Mat>{ bestBGRAImage },
			vector<cv::Mat>{ bestBGRImage, bestAImage }, { 0, 0, 1, 1, 2, 2, 3, 3 });
		cv::Mat bestHSVImage;
		cv::cvtColor(bestBGRImage, bestHSVImage, cv::COLOR_BGR2HSV);
		cv::Mat bestVImage(bestHSVImage.size(), CV_8UC1);
		cv::mixChannels(vector<cv::Mat>{ bestHSVImage }, vector<cv::Mat>{ bestVImage }, { 2, 0 });
		bestVImage = util::equalizeHistMask(bestVImage, bestAImage);
		cv::mixChannels(vector<cv::Mat>{ bestVImage }, vector<cv::Mat> { bestHSVImage }, { 0, 2 });
		cv::cvtColor(bestHSVImage, bestBGRImage, cv::COLOR_HSV2BGR);
		cv::mixChannels(vector<cv::Mat>{ bestBGRImage, bestAImage },
			vector<cv::Mat>{ bestBGRAImage }, { 0, 0, 1, 1, 2, 2, 3, 3 });
		// Save histeq version
		fs::path bestHisteqPath = imageHisteqDir / (cluster + "_" + fiStr + ".png");
		cv::imwrite(bestHisteqPath.string(), bestBGRAImage);

		// Create JSON metadata
		rj::Document meta;
		meta.SetObject();
		auto& alloc = meta.GetAllocator();
		meta.AddMember("region", rj::StringRef(region.c_str()), alloc);
		meta.AddMember("cluster", rj::StringRef(cluster.c_str()), alloc);
		meta.AddMember("model", rj::StringRef(model.c_str()), alloc);
		meta.AddMember("facade", rj::Value().SetUint(fi), alloc);
		meta.AddMember("satellite", rj::StringRef(maxScoreIdx.c_str()), alloc);
		meta.AddMember("roof", rj::Value().SetBool(finfo.roof), alloc);
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
		meta.AddMember("imagename", rj::Value().SetString(chipPath.string().c_str(), alloc).Move(), alloc);

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

// Synthesize facade geometry using deep network parameters
void Building::synthFacadeGeometry(fs::path outputDir, map<size_t, fs::path> facades) {

	// Initialize OpenGL
	OpenGLContext ctx;
	glClearColor(0.0, 0.0, 0.0, 0.0);
	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
	glPixelStorei(GL_PACK_ALIGNMENT, 1);

	// Create framebuffer texture
	GLuint fbtex = ctx.genTexture();
	glBindTexture(GL_TEXTURE_2D, fbtex);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 0, 0, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glBindTexture(GL_TEXTURE_2D, 0);
	// Create framebuffer object
	GLuint fbo = ctx.genFramebuffer();
	glBindFramebuffer(GL_FRAMEBUFFER, fbo);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, fbtex, 0);

	// Create position buffer
	GLuint pbuf = ctx.genBuffer();
	glBindBuffer(GL_ARRAY_BUFFER, pbuf);
	glBufferData(GL_ARRAY_BUFFER, 0, NULL, GL_STATIC_DRAW);
	// Create color buffer
	GLuint cbuf = ctx.genBuffer();
	glBindBuffer(GL_ARRAY_BUFFER, cbuf);
	glBufferData(GL_ARRAY_BUFFER, 0, NULL, GL_STATIC_DRAW);

	// Create vertex array object
	GLuint vao = ctx.genVAO();
	glBindVertexArray(vao);
	glBindBuffer(GL_ARRAY_BUFFER, pbuf);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(glm::vec2), 0);
	glBindBuffer(GL_ARRAY_BUFFER, cbuf);
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), 0);

	// Vertex shader
	static const string vsh_str = R"(
		#version 460

		layout(location = 0) in vec2 pos;
		layout(location = 1) in vec3 color;

		smooth out vec3 fragCol;

		uniform mat4 xform;

		void main() {
			gl_Position = xform * vec4(pos, 0.0, 1.0);
			fragCol = color;
		})";
	// Fragment shader
	static const string fsh_str = R"(
		#version 460

		smooth in vec3 fragCol;

		out vec4 outCol;

		void main() {
			outCol = vec4(fragCol, 1.0);
		})";

	// Compile and link shaders
	vector<GLuint> shaders;
	shaders.push_back(ctx.compileShader(GL_VERTEX_SHADER, vsh_str));
	shaders.push_back(ctx.compileShader(GL_FRAGMENT_SHADER, fsh_str));
	GLuint program = ctx.linkProgram(shaders);
	GLuint xformLoc = glGetUniformLocation(program, "xform");
	glUseProgram(program);


	fs::path outDir = outputDir / region / model / cluster;
	// Create synth facade directory
	fs::path synthDir = outDir / "synth";
	if (fs::exists(synthDir))
		fs::remove_all(synthDir);
	fs::create_directory(synthDir);

	// Create output OBJ and MTL files
	fs::path objPath = outDir / (cluster + "_synth_geometry.obj");
	fs::path mtlPath = objPath; mtlPath.replace_extension(".mtl");
	ofstream objFile(objPath);
	ofstream mtlFile(mtlPath);
	int vcount = 0;
	int ncount = 0;
	int tcount = 0;
	float recess = 0.0;		// Amount to recess windows and doors into the building
	float g34_border = 2.0;	// Border above and below vertical windows

	// Add material for texture-mapped roofs
	mtlFile << "newmtl atlas" << endl;
	mtlFile << "map_Kd " << cluster << "_synth_atlas.png" << endl;

	objFile << "mtllib " << mtlPath.filename().string() << endl;

	// Write up and down normals for window recesses
	objFile << "vn 0.0 0.0 1.0" << endl;
	objFile << "vn 0.0 0.0 -1.0" << endl;
	int normUIdx = ++ncount;	// Index of up normal
	int normDIdx = ++ncount;	// Index of down normal

	// Vertical sections in which windows are placed
	struct WinSection {
		glm::vec2 minBB;		// Minimum rectified 2D coords of section
		glm::vec2 maxBB;		// Maximum rectified 2D coords of section
		int rows;				// Number of rows of windows
		int cols;				// Number of columns of windows
		float xscale;			// Horizontal scale factor for window spacing
		float yoffset;			// Vertical offset to align rows
	};

	// Store facade parameters
	struct FacadeParams {
		bool roof;				// Whether this facade is a roof
		float score;			// Score of the facade
		string imagename;		// Path to highest scoring facade chip
		glm::vec3 norm;			// Facade normal
		glm::mat4 xform;		// Transforms from 3D to rectified 2D
		glm::mat4 invXform;		// Transforms from rectified 2D to 3D
		glm::vec2 geom_size;	// Width and height after applied xform
		vector<WinSection> winSections;		// Vertical window sections

		bool valid;					// Facade has valid parameters
		glm::vec2 chip_size;		// Size of chip given to DN, in meters
		glm::vec3 bg_color;			// Background color (RGB, [0, 1])
		glm::vec3 window_color;		// Window/door color (RGB, [0, 1])
		int grammar;				// Which grammar this facade uses
		int rows;					// Number of window rows per chip height
		int cols;					// Number of window columns per chip width
		float relativeWidth;		// Width of window relative to window cell
		float relativeHeight;		// Height of window relative to window cell
		bool hasDoors;				// Whether we have doors
		int doors;					// Number of doors per chip width
		float relativeDWidth;		// Width of doors relative to door cell
		float relativeDHeight;		// Height of doors relative to chip height
	};
	map<size_t, FacadeParams> facadeParams;

	// "Fuzzy" height comparator
	auto heightCmp = [](const float& a, const float& b) -> bool {
		static const float eps = 1e-3;
		if (abs(a - b) > eps) return a < b;
		return false;
	};
	// Group facades with similar parameters together
	struct FacadeGroup {
		vector<size_t> facades;		// Which facades are in the group
		vector<float> grammars;		// Which grammars are represented and by how much
		set<float, decltype(heightCmp)> sheights;	// Heights of winsections in this group

		bool valid;					// Whether group has valid parameters
		int grammar;				// The selected grammar

		float avgRowsPerMeter;		// Average parameter values
		float avgColsPerMeter;
		float avgRelWidth;
		float avgRelHeight;
		bool hasDoors;
		float avgDoorsPerMeter;
		float avgRelDWidth;
		float avgDHeight;			// Not relative D height; chip size differs

		FacadeGroup(decltype(heightCmp) cmp) : grammars(7, 0.0), sheights(cmp),
			valid(false), grammar(0), avgRowsPerMeter(0.0), avgColsPerMeter(0.0),
			avgRelWidth(0.0), avgRelHeight(0.0), hasDoors(false), avgDoorsPerMeter(0.0),
			avgRelDWidth(0.0), avgDHeight(0.0) {}
	};
	vector<FacadeGroup> facadeGroups;
	vector<int> whichGroup(facadeInfo.size(), -1);

	for (size_t fi = 0; fi < facadeInfo.size(); fi++) {
		// Skip if no metadata on this facade
		if (!facades.count(fi)) continue;

		// Read metadata
		ifstream metaFile(facades[fi]);
		rj::IStreamWrapper isw(metaFile);
		rj::Document meta;
		meta.ParseStream(isw);

		// Store parameters from metadata
		FacadeParams fp;
		fp.roof = meta["roof"].GetBool();
		fp.score = meta["score"].GetFloat();
		fp.imagename = meta["imagename"].GetString();

		if (meta.HasMember("valid")) {
			fp.valid = meta["valid"].GetBool();
			fp.bg_color.x = meta["bg_color"][2].GetDouble() / 255;
			fp.bg_color.y = meta["bg_color"][1].GetDouble() / 255;
			fp.bg_color.z = meta["bg_color"][0].GetDouble() / 255;
		} else {
			fp.valid = false;
			fp.bg_color = { 0.0, 0.0, 0.0 };
		}

		fp.window_color = { 0.0, 0.0, 0.0 };
		if (fp.valid) {
			fp.chip_size.x = meta["chip_size"][0].GetDouble();
			fp.chip_size.y = meta["chip_size"][1].GetDouble();
			fp.window_color.x = meta["window_color"][2].GetDouble() / 255;
			fp.window_color.y = meta["window_color"][1].GetDouble() / 255;
			fp.window_color.z = meta["window_color"][0].GetDouble() / 255;
			fp.grammar = meta["grammar"].GetInt();
			fp.rows = meta["paras"]["rows"].GetInt();
			fp.cols = meta["paras"]["cols"].GetInt();
			fp.relativeWidth = meta["paras"]["relativeWidth"].GetFloat();
			fp.relativeHeight = meta["paras"]["relativeHeight"].GetFloat();
			fp.hasDoors = meta["paras"].HasMember("doors");
			fp.doors = 0;
			fp.relativeDWidth = 0.0;
			fp.relativeDHeight = 0.0;
			if (fp.hasDoors) {
				fp.doors = meta["paras"]["doors"].GetInt();
				fp.relativeDWidth = meta["paras"]["relativeDWidth"].GetFloat();
				fp.relativeDHeight = meta["paras"]["relativeDHeight"].GetFloat();
			}
		} else {
			fp.grammar = 0;
			fp.hasDoors = false;
		}

		// Reorient facade for easier window placement
		fp.xform = glm::mat4(1.0);
		fp.norm = glm::normalize(facadeInfo[fi].normal);
		if (glm::dot(fp.norm, { 0.0, 0.0, 1.0 }) < 1.0) {
			glm::vec3 up(0.0, 0.0, 1.0);
			glm::vec3 right = glm::normalize(glm::cross(up, fp.norm));
			up = glm::normalize(glm::cross(fp.norm, right));

			fp.xform[0] = glm::vec4(right, 0.0f);
			fp.xform[1] = glm::vec4(up, 0.0f);
			fp.xform[2] = glm::vec4(fp.norm, 0.0f);
			fp.xform = glm::transpose(fp.xform);
		}

		// Get rotated facade offset
		glm::vec3 minXYZ(FLT_MAX);
		fp.geom_size = glm::vec2(-FLT_MAX);
		for (auto f : facadeInfo[fi].faceIDs) {
			for (int vi = 0; vi < 3; vi++) {
				glm::vec3 v = posBuf[indexBuf[3 * f + vi]];
				minXYZ = glm::min(minXYZ, glm::vec3(fp.xform * glm::vec4(v, 1.0)));
				fp.geom_size = glm::max(fp.geom_size, glm::vec2(fp.xform * glm::vec4(v, 1.0)));
			}
		}
		fp.geom_size -= glm::vec2(minXYZ);
		fp.xform[3] = glm::vec4(-minXYZ, 1.0);
		fp.invXform = glm::inverse(fp.xform);

		if (!fp.roof) {
			// Get section boundaries along X
			auto sepCmp = [](const float& a, const float& b) -> bool {
				static const float eps = 1e-2;
				if (abs(a - b) > eps) return a < b;
				return false;
			};
			set<float, decltype(sepCmp)> xsep(sepCmp);
			for (auto f : facadeInfo[fi].faceIDs) {
				for (int vi = 0; vi < 3; vi++) {
					glm::vec3 v = posBuf[indexBuf[3 * f + vi]];
					xsep.insert((fp.xform * glm::vec4(v, 1.0)).x);
				}
			}

			// Separate facade into window sections
			auto& winSections = fp.winSections;
			for (auto xi = xsep.begin(); xi != xsep.end(); ++xi) {
				if (!winSections.empty())
					winSections.back().maxBB.x = *xi;
				auto xiNext = xi; ++xiNext;
				if (xiNext != xsep.end()) {
					winSections.push_back({});
					winSections.back().minBB.x = *xi;
				}
			}

			// Get vertical bounds of each section
			for (auto f : facadeInfo[fi].faceIDs) {
				// Get triangle bbox
				glm::vec2 minTBB(FLT_MAX);
				glm::vec2 maxTBB(-FLT_MAX);
				for (int vi = 0; vi < 3; vi++) {
					glm::vec3 v = posBuf[indexBuf[3 * f + vi]];
					minTBB = glm::min(minTBB, glm::vec2(fp.xform * glm::vec4(v, 1.0)));
					maxTBB = glm::max(maxTBB, glm::vec2(fp.xform * glm::vec4(v, 1.0)));
				}

				// Intersect with all sections
				for (auto& s : winSections) {
					if (minTBB.x + 1e-2 < s.maxBB.x && maxTBB.x - 1e-2 > s.minBB.x) {
						s.minBB.y = min(s.minBB.y, minTBB.y);
						s.maxBB.y = max(s.maxBB.y, maxTBB.y);
					}
				}
			}

			// Combine adjacent sections of equal vertical bounds
			for (auto si = winSections.begin(); si != winSections.end();) {
				auto siNext = si; siNext++;
				if (siNext == winSections.end()) break;
				if (abs(si->minBB.y - siNext->minBB.y) < 1e-4 &&
					abs(si->maxBB.y - siNext->maxBB.y) < 1e-4) {
					si->maxBB.x = siNext->maxBB.x;
					winSections.erase(siNext);
				} else
					++si;
			}
		}

		// Store parameters for this facade
		facadeParams[fi] = fp;

		if (fp.roof) continue;

		// Get tallest two sections
		vector<float> tallestSections;
		for (auto si : fp.winSections)
			tallestSections.push_back(si.maxBB.y - si.minBB.y);
		sort(tallestSections.begin(), tallestSections.end(), [](float a, float b) -> bool {
			return b < a; });
		tallestSections.resize(1);


		// Find a group for this facade
		bool inGroup = false;
		for (int g = 0; g < facadeGroups.size(); g++) {
			FacadeGroup& fg = facadeGroups[g];

			// Find a matching section height
			bool foundSection = false;
			for (auto ts : tallestSections) {
				if (fg.sheights.count(ts))
					foundSection = true;
			}

			// Add this facade and its grammar and heights to the group
			if (foundSection) {
				fg.facades.push_back(fi);

				if (fp.valid)
					fg.grammars[fp.grammar] = max(fg.grammars[fp.grammar], fp.score);
				else
					fg.grammars[0] = max(fg.grammars[0], fp.score);

				for (auto ts : tallestSections)
					fg.sheights.insert(ts);

				whichGroup[fi] = g;
				inGroup = true;
				break;
			}
		}
		// If no group matched, add a new group
		if (!inGroup) {
			FacadeGroup fg(heightCmp);

			fg.facades.push_back(fi);

			if (fp.valid)
				fg.grammars[fp.grammar] += fp.score;
			else
				fg.grammars[0] += fp.score;

			for (auto ts : tallestSections)
				fg.sheights.insert(ts);

			whichGroup[fi] = facadeGroups.size();
			facadeGroups.push_back(fg);
		}
	}
	// Determine group grammar and parameters
	for (auto& fg : facadeGroups) {
		// Get the grammar with the highest total score
		float maxG = 0.1;
		fg.grammar = 0;
		for (int g = 1; g < fg.grammars.size(); g++)
			if (fg.grammars[g] > maxG) {
				maxG = fg.grammars[g];
				fg.grammar = g;
			}
		// If any in this group have doors, add doors
		if (fg.grammar == 1 || fg.grammar == 3 || fg.grammar == 5)
			if (fg.grammars[2] > 0 || fg.grammars[4] > 0 || fg.grammars[6] > 0)
				fg.grammar++;
		fg.hasDoors = (fg.grammar == 2 || fg.grammar == 4 || fg.grammar == 6);

		// Group is valid if grammar is non-zero
		fg.valid = (fg.grammar > 0);
		// Set facades to valid or invalid
		for (auto fi : fg.facades)
			facadeParams[fi].valid = fg.valid;

		if (!fg.valid) continue;

		// Average out the parameter values for selected grammar
		int sz = 0;
		int szd = 0;
		for (auto fi : fg.facades) {
			auto& fp = facadeParams[fi];
			if (fp.grammar && (fp.grammar - 1) / 2 == (fg.grammar - 1) / 2) {
				sz++;
				if (fp.hasDoors)
					fg.avgRowsPerMeter += fp.rows / (fp.chip_size.y * (1.0 - fp.relativeDHeight));
				else
					fg.avgRowsPerMeter += fp.rows / fp.chip_size.y;
				fg.avgColsPerMeter += fp.cols / fp.chip_size.x;
				fg.avgRelWidth += fp.relativeWidth;
				fg.avgRelHeight += fp.relativeHeight;
			}
			if (fp.grammar && fp.hasDoors) {
				szd++;
				fg.avgDoorsPerMeter += fp.doors / fp.chip_size.x;
				fg.avgRelDWidth += fp.relativeDWidth;
				fg.avgDHeight += fp.relativeDHeight * fp.chip_size.y;
			}
		}
		fg.avgRowsPerMeter /= sz;
		fg.avgColsPerMeter /= sz;
		fg.avgRelWidth /= sz;
		fg.avgRelHeight /= sz;
		if (szd) {
			fg.avgDoorsPerMeter /= szd;
			fg.avgRelDWidth /= szd;
			fg.avgDHeight /= szd;
		}
	}

	// Iterate over all facades
	for (size_t fi = 0; fi < facadeInfo.size(); fi++) {
		string fiStr; {
			stringstream ss;
			ss << setw(4) << setfill('0') << fi;
			fiStr = ss.str();
		}

		// If we have DN metadata for this facade, synthesize a facade with windows
		if (facadeParams.count(fi)) {
			FacadeParams& fp = facadeParams[fi];

			// If valid DN output, add windows and doors
			if (fp.valid) {
				assert(whichGroup[fi] >= 0);
				const FacadeGroup& fg = facadeGroups[whichGroup[fi]];

				// Get sizes and spacing
				float winCellW = 1.0 / fg.avgColsPerMeter;
				float winCellH = 1.0 / fg.avgRowsPerMeter;
				float winW = winCellW * fg.avgRelWidth;
				float winH = winCellH * fg.avgRelHeight;
				float winXsep = winCellW * (1.0 - fg.avgRelWidth);
				float winYsep = winCellH * (1.0 - fg.avgRelHeight);
				float winXoff = winXsep / 2.0;
				float winYoff = winYsep / 2.0;
				float doorCellW = 1.0 / max(fg.avgDoorsPerMeter, 0.01f);
				float doorW = doorCellW * fg.avgRelDWidth;
				float doorH = fg.avgDHeight;
				float doorXsep = doorCellW * (1.0 - fg.avgRelDWidth);
				float doorXoff = doorXsep / 2.0;

				auto& xform = fp.xform;
				auto& invXform = fp.invXform;
				auto& winSections = fp.winSections;
				// Separate window sections into door sections if we have any doors
				struct DoorSection {
					glm::vec2 minBB;
					glm::vec2 maxBB;
					int cols;
					float xoffset;
				};
				vector<DoorSection> doorSections;
				if (fg.hasDoors) {
					// Iterate over window sections
					for (auto wi = winSections.begin(); wi != winSections.end();) {
						// Win section is entirely below door line
						if (wi->maxBB.y < doorH) {
							doorSections.push_back({});
							doorSections.back().minBB = wi->minBB;
							doorSections.back().maxBB = wi->maxBB;
							wi = winSections.erase(wi);

						// Win section is partially below door line
						} else if (wi->minBB.y < doorH) {
							doorSections.push_back({});
							doorSections.back().minBB = wi->minBB;
							doorSections.back().maxBB.x = wi->maxBB.x;
							doorSections.back().maxBB.y = doorH;
							wi->minBB.y = doorH;
							++wi;

						// Win section is completely above door line
						} else
							++wi;
					}

					// Combine adjacent door sections of equal vertical bounds
					for (auto di = doorSections.begin(); di != doorSections.end();) {
						auto diNext = di; ++diNext;
						if (diNext == doorSections.end()) break;
						if (abs(di->minBB.y - diNext->minBB.y) < 1e-4 &&
							abs(di->maxBB.y - diNext->maxBB.y) < 1e-4) {
							di->maxBB.x = diNext->maxBB.x;
							doorSections.erase(diNext);
						} else
							++di;
					}
				}

				// Position and color buffers
				vector<glm::vec2> synthPosBuf;
				vector<glm::vec3> synthColBuf;

				// Method to write a face to the OBJ file
				auto writeFace = [&](glm::vec3 va, glm::vec3 vb, glm::vec3 vc, glm::vec3 vd,
					int nidx, bool window) {

					// Set the color to use
					glm::vec3 color = window ? fp.window_color : fp.bg_color;

					// Add verts to position buffer
					synthPosBuf.push_back(glm::vec2(va));
					synthPosBuf.push_back(glm::vec2(vb));
					synthPosBuf.push_back(glm::vec2(vc));
					synthPosBuf.push_back(glm::vec2(vc));
					synthPosBuf.push_back(glm::vec2(vd));
					synthPosBuf.push_back(glm::vec2(va));
					// Add colors to color buffer
					synthColBuf.push_back(color);
					synthColBuf.push_back(color);
					synthColBuf.push_back(color);
					synthColBuf.push_back(color);
					synthColBuf.push_back(color);
					synthColBuf.push_back(color);

					// Transform positions
					va = glm::vec3(invXform * glm::vec4(va, 1.0));
					vb = glm::vec3(invXform * glm::vec4(vb, 1.0));
					vc = glm::vec3(invXform * glm::vec4(vc, 1.0));
					vd = glm::vec3(invXform * glm::vec4(vd, 1.0));

					// Write positions
					objFile << "v " << va.x << " " << va.y << " " << va.z << " "
						<< color.x << " " << color.y << " " << color.z << endl;
					objFile << "v " << vb.x << " " << vb.y << " " << vb.z << " "
						<< color.x << " " << color.y << " " << color.z << endl;
					objFile << "v " << vc.x << " " << vc.y << " " << vc.z << " "
						<< color.x << " " << color.y << " " << color.z << endl;
					objFile << "v " << vd.x << " " << vd.y << " " << vd.z << " "
						<< color.x << " " << color.y << " " << color.z << endl;

					// Write indices
					objFile << "f " << vcount+1 << "//" << nidx << " "
						<< vcount+2 << "//" << nidx << " "
						<< vcount+3 << "//" << nidx << endl;
					objFile << "f " << vcount+3 << "//" << nidx << " "
						<< vcount+4 << "//" << nidx << " "
						<< vcount+1 << "//" << nidx << endl;

					vcount += 4;
				};

				// Add materials for window and background
				mtlFile << "newmtl " << fiStr << "_bg" << endl;
				mtlFile << "Kd " << fp.bg_color.x << " "
								<< fp.bg_color.y << " "
								<< fp.bg_color.z << endl;
				mtlFile << "newmtl " << fiStr << "_window" << endl;
				mtlFile << "Kd " << fp.window_color.x << " "
								<< fp.window_color.y << " "
								<< fp.window_color.z << endl;

				objFile << "usemtl " << fiStr << "_bg" << endl;

				// Add facade normals
				glm::vec3 normR = glm::normalize(glm::cross(glm::vec3(0.0, 0.0, 1.0), fp.norm));
				glm::vec3 normL = -normR;
				objFile << "vn " << fp.norm.x << " " << fp.norm.y << " " << fp.norm.z << endl;
				objFile << "vn " << normR.x << " " << normR.y << " " << normR.z << endl;
				objFile << "vn " << normL.x << " " << normL.y << " " << normL.z << endl;
				int normIdx = ++ncount;
				int normRIdx = ++ncount;
				int normLIdx = ++ncount;

				// Center doors on each door section
				for (auto& d : doorSections) {
					if (d.maxBB.y - d.minBB.y < doorH) {
						d.cols = 0;
					} else {
						d.cols = floor((d.maxBB.x - d.minBB.x + doorXsep / 2) / doorCellW);
						d.xoffset = ((d.maxBB.x - d.minBB.x) - d.cols * doorCellW) / 2;
					}

					// If no doors, just output the segment
					if (d.cols == 0) {
						glm::vec3 va(d.minBB.x, d.minBB.y, 0.0);
						glm::vec3 vb(d.maxBB.x, d.minBB.y, 0.0);
						glm::vec3 vc(d.maxBB.x, d.maxBB.y, 0.0);
						glm::vec3 vd(d.minBB.x, d.maxBB.y, 0.0);
						writeFace(va, vb, vc, vd, normIdx, false);
						continue;
					}

					for (int c = 0; c < d.cols; c++) {
						float dMinX = d.minBB.x + d.xoffset + doorXoff + c * doorCellW;
						float dMaxX = dMinX + doorW;

						// If first door, write spacing to left side of section
						if (c == 0) {
							glm::vec3 va(d.minBB.x, d.minBB.y, 0.0);
							glm::vec3 vb(dMinX, d.minBB.y, 0.0);
							glm::vec3 vc(dMinX, d.maxBB.y, 0.0);
							glm::vec3 vd(d.minBB.x, d.maxBB.y, 0.0);
							writeFace(va, vb, vc, vd, normIdx, false);
						// Otherwise write the spacing before this door
						} else {
							glm::vec3 va(dMinX - doorXsep, d.minBB.y, 0.0);
							glm::vec3 vb(dMinX, d.minBB.y, 0.0);
							glm::vec3 vc(dMinX, d.maxBB.y, 0.0);
							glm::vec3 vd(dMinX - doorXsep, d.maxBB.y, 0.0);
							writeFace(va, vb, vc, vd, normIdx, false);
						}

						// Get door vertices
						glm::vec3 va(dMinX, d.minBB.y, 0.0);
						glm::vec3 vb(dMaxX, d.minBB.y, 0.0);
						glm::vec3 vc(dMaxX, d.maxBB.y, 0.0);
						glm::vec3 vd(dMinX, d.maxBB.y, 0.0);
						glm::vec3 va2(dMinX, d.minBB.y, -recess);
						glm::vec3 vb2(dMaxX, d.minBB.y, -recess);
						glm::vec3 vc2(dMaxX, d.maxBB.y, -recess);
						glm::vec3 vd2(dMinX, d.maxBB.y, -recess);
						// Write the door boundaries
						if (recess > 0.0) {
							writeFace(vd2, vc2, vc, vd, normDIdx, false);
							writeFace(va, va2, vd2, vd, normRIdx, false);
							writeFace(vb2, vb, vc, vc2, normLIdx, false);
						}
						// Write the door face
						objFile << "usemtl " << fiStr << "_window" << endl;
						writeFace(va2, vb2, vc2, vd2, normIdx, true);
						objFile << "usemtl " << fiStr << "_bg" << endl;

						// If last door, also write spacing to right side of section
						if (c+1 == d.cols) {
							glm::vec3 va(dMaxX, d.minBB.y, 0.0);
							glm::vec3 vb(d.maxBB.x, d.minBB.y, 0.0);
							glm::vec3 vc(d.maxBB.x, d.maxBB.y, 0.0);
							glm::vec3 vd(dMaxX, d.maxBB.y, 0.0);
							writeFace(va, vb, vc, vd, normIdx, false);
						}
					}
				}

				// Skip if all window sections became door sections
				if (winSections.empty()) continue;

				// Get the tallest window section
				int maxH = -1;
				for (int si = 0; si < winSections.size(); ++si) {
					if (maxH < 0 || (winSections[si].maxBB.y - winSections[si].minBB.y) >
						(winSections[maxH].maxBB.y - winSections[maxH].minBB.y))
						maxH = si;
				}
				if (fg.grammar != 3 && fg.grammar != 4) {
					// Center windows vertically on tallest window section
					winSections[maxH].rows = floor(
						(winSections[maxH].maxBB.y - winSections[maxH].minBB.y) / winCellH);
					winSections[maxH].yoffset =
						((winSections[maxH].maxBB.y - winSections[maxH].minBB.y) -
						winSections[maxH].rows * winCellH) / 2;
				}

				// Output all windows
				for (int si = 0; si < winSections.size(); si++) {
					WinSection& s = winSections[si];
					if (fg.grammar != 5 && fg.grammar != 6) {
						// Stretch spacing between columns horizontally on all sections
						s.cols = floor((s.maxBB.x - s.minBB.x) / winCellW);
						s.xscale = (s.cols == 0) ? 1.0 : (s.maxBB.x - s.minBB.x) / (s.cols * winCellW);
					} else {
						// If horizontal windows, only use one column
						s.cols = 1;
						s.xscale = 1.0;
					}

					if (fg.grammar != 3 && fg.grammar != 4) {
						// Align rows with rows on the tallest section
						if (si != maxH) {
							const WinSection& sm = winSections[maxH];
							s.yoffset = ceil((s.minBB.y - sm.minBB.y - sm.yoffset) / winCellH)
								* winCellH + sm.minBB.y + sm.yoffset - s.minBB.y;
							s.rows = floor((s.maxBB.y - s.minBB.y - s.yoffset) / winCellH);
						}
					} else {
						// If vertical windows, only use one row (if section big enough)
						if (s.maxBB.y - s.minBB.y > 2 * g34_border) {
							s.rows = 1;
							s.yoffset = 0.0;
						} else {
							s.rows = 0;
							s.yoffset = 0.0;
						}
					}

					// If no rows or columns, just output the segment
					if (s.rows <= 0 || s.cols <= 0) {
						glm::vec3 va(s.minBB.x, s.minBB.y, 0.0);
						glm::vec3 vb(s.maxBB.x, s.minBB.y, 0.0);
						glm::vec3 vc(s.maxBB.x, s.maxBB.y, 0.0);
						glm::vec3 vd(s.minBB.x, s.maxBB.y, 0.0);
						writeFace(va, vb, vc, vd, normIdx, false);
						continue;
					}

					for (int r = 0; r < s.rows; r++) {
						// Get spacing for special cases (vertical windows)
						float winYoff_s = winYoff;
						float winCellH_s = winCellH;
						float winH_s = winH;
						if (fg.grammar == 3 || fg.grammar == 4) {
							winCellH_s = s.maxBB.y - s.minBB.y;
							winYoff_s = g34_border;
							winH_s = winCellH_s - 2 * winYoff_s;
						}

						float wMinY = s.minBB.y + s.yoffset + winYoff_s + r * winCellH_s;
						float wMaxY = wMinY + winH_s;

						// If first row, write spacing below all windows
						if (r == 0) {
							glm::vec3 va(s.minBB.x, s.minBB.y, 0.0);
							glm::vec3 vb(s.maxBB.x, s.minBB.y, 0.0);
							glm::vec3 vc(s.maxBB.x, wMinY, 0.0);
							glm::vec3 vd(s.minBB.x, wMinY, 0.0);
							writeFace(va, vb, vc, vd, normIdx, false);
						// Otherwise, write spacing between rows
						} else {
							glm::vec3 va(s.minBB.x, wMinY - winYsep, 0.0);
							glm::vec3 vb(s.maxBB.x, wMinY - winYsep, 0.0);
							glm::vec3 vc(s.maxBB.x, wMinY, 0.0);
							glm::vec3 vd(s.minBB.x, wMinY, 0.0);
							writeFace(va, vb, vc, vd, normIdx, false);
						}

						// Write all windows in this row
						for (int c = 0; c < s.cols; c++) {
							float wXsep = winCellW * s.xscale - winW;
							float wMinX = s.minBB.x + wXsep / 2 + c * winCellW * s.xscale;
							float wMaxX = wMinX + winW;

							// Get spacing for special cases (horizontal windows)
							if (fg.grammar == 5 || fg.grammar == 6) {
								wMinX = s.minBB.x;
								wMaxX = s.maxBB.x;
							}

							if (fg.grammar != 5 && fg.grammar != 6) {
								// If first window, write spacing to the left of the row
								if (c == 0) {
									glm::vec3 va(s.minBB.x, wMinY, 0.0);
									glm::vec3 vb(wMinX, wMinY, 0.0);
									glm::vec3 vc(wMinX, wMaxY, 0.0);
									glm::vec3 vd(s.minBB.x, wMaxY, 0.0);
									writeFace(va, vb, vc, vd, normIdx, false);
								// Otherwise, write spacing between columns
								} else {
									glm::vec3 va(wMinX - wXsep, wMinY, 0.0);
									glm::vec3 vb(wMinX, wMinY, 0.0);
									glm::vec3 vc(wMinX, wMaxY, 0.0);
									glm::vec3 vd(wMinX - wXsep, wMaxY, 0.0);
									writeFace(va, vb, vc, vd, normIdx, false);
								}
							}

							// Get the window vertices
							glm::vec3 va(wMinX, wMinY, 0.0);
							glm::vec3 vb(wMaxX, wMinY, 0.0);
							glm::vec3 vc(wMaxX, wMaxY, 0.0);
							glm::vec3 vd(wMinX, wMaxY, 0.0);
							glm::vec3 va2(wMinX, wMinY, -recess);
							glm::vec3 vb2(wMaxX, wMinY, -recess);
							glm::vec3 vc2(wMaxX, wMaxY, -recess);
							glm::vec3 vd2(wMinX, wMaxY, -recess);
							// Write the window boundaries
							if (recess > 0.0) {
								writeFace(va, vb, vb2, va2, normUIdx, false);
								writeFace(vd2, vc2, vc, vd, normDIdx, false);
								writeFace(va, va2, vd2, vd, normRIdx, false);
								writeFace(vb2, vb, vc, vc2, normLIdx, false);
							}
							// Write the window in this row/column
							objFile << "usemtl " << fiStr << "_window" << endl;
							writeFace(va2, vb2, vc2, vd2, normIdx, true);
							objFile << "usemtl " << fiStr << "_bg" << endl;

							if (fg.grammar != 5 && fg.grammar != 6) {
								// If the last window, write spacing to the right of the row
								if (c+1 == s.cols) {
									glm::vec3 va(wMaxX, wMinY, 0.0);
									glm::vec3 vb(s.maxBB.x, wMinY, 0.0);
									glm::vec3 vc(s.maxBB.x, wMaxY, 0.0);
									glm::vec3 vd(wMaxX, wMaxY, 0.0);
									writeFace(va, vb, vc, vd, normIdx, false);
								}
							}
						}

						// If last row, write spacing above all windows
						if (r+1 == s.rows) {
							glm::vec3 va(s.minBB.x, wMaxY, 0.0);
							glm::vec3 vb(s.maxBB.x, wMaxY, 0.0);
							glm::vec3 vc(s.maxBB.x, s.maxBB.y, 0.0);
							glm::vec3 vd(s.minBB.x, s.maxBB.y, 0.0);
							writeFace(va, vb, vc, vd, normIdx, false);
						}
					}
				}

				// Upload positions and colors
				glBindBuffer(GL_ARRAY_BUFFER, pbuf);
				glBufferData(GL_ARRAY_BUFFER, synthPosBuf.size() * sizeof(synthPosBuf[0]),
					synthPosBuf.data(), GL_STATIC_DRAW);
				glBindBuffer(GL_ARRAY_BUFFER, cbuf);
				glBufferData(GL_ARRAY_BUFFER, synthColBuf.size() * sizeof(synthColBuf[0]),
					synthColBuf.data(), GL_STATIC_DRAW);

				// Upload xform matrix
				glm::mat4 glXform(1.0);
				glXform[0][0] = 2.0 / fp.geom_size.x;
				glXform[1][1] = 2.0 / fp.geom_size.y;
				glXform[3][0] = -1.0;
				glXform[3][1] = -1.0;
				glUniformMatrix4fv(xformLoc, 1, GL_FALSE, glm::value_ptr(glXform));

				// Resize framebuffer texture
				glm::uvec2 synthSizePx = facadeInfo[fi].size;
				glBindTexture(GL_TEXTURE_2D, fbtex);
				glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, synthSizePx.x, synthSizePx.y, 0,
					GL_RGBA, GL_UNSIGNED_BYTE, NULL);
				glBindTexture(GL_TEXTURE_2D, 0);
				glViewport(0, 0, synthSizePx.x, synthSizePx.y);
				glClear(GL_COLOR_BUFFER_BIT);

				// Draw facade geometry onto texture
				glDrawArrays(GL_TRIANGLES, 0, synthPosBuf.size());

				// Download synth texture
				cv::Mat synthImage(synthSizePx.y, synthSizePx.x, CV_8UC4);
				glBindTexture(GL_TEXTURE_2D, fbtex);
				glGetTexImage(GL_TEXTURE_2D, 0, GL_BGRA, GL_UNSIGNED_BYTE, synthImage.data);
				cv::flip(synthImage, synthImage, 0);

				// Save to disk
				fs::path synthPath = synthDir / (cluster + "_" + fiStr + ".png");
				cv::imwrite(synthPath.string(), synthImage);


			// If it's a roof, add existing texture-mapped geometry
			} else if (fp.roof) {
				// Use texture-mapped material
				objFile << "usemtl atlas" << endl;

				// Write the normal
				glm::vec3 norm = glm::normalize(facadeInfo[fi].normal);
				objFile << "vn " << norm.x << " " << norm.y << " " << norm.z << endl;
				int normIdx = ++ncount;

				// Add each triangle
				for (auto f : facadeInfo[fi].faceIDs) {
					// Write positions
					glm::vec3 va = posBuf[indexBuf[3 * f + 0]];
					glm::vec3 vb = posBuf[indexBuf[3 * f + 1]];
					glm::vec3 vc = posBuf[indexBuf[3 * f + 2]];
					objFile << "v " << va.x << " " << va.y << " " << va.z << endl;
					objFile << "v " << vb.x << " " << vb.y << " " << vb.z << endl;
					objFile << "v " << vc.x << " " << vc.y << " " << vc.z << endl;

					// Write texture coords
					glm::vec2 ta = atlasTCBuf[indexBuf[3 * f + 0]];
					glm::vec2 tb = atlasTCBuf[indexBuf[3 * f + 1]];
					glm::vec2 tc = atlasTCBuf[indexBuf[3 * f + 2]];
					objFile << "vt " << ta.x << " " << ta.y << endl;
					objFile << "vt " << tb.x << " " << tb.y << endl;
					objFile << "vt " << tc.x << " " << tc.y << endl;

					// Write indices
					objFile << "f " << vcount+1 << "/" << tcount+1 << "/" << normIdx << " "
						<< vcount+2 << "/" << tcount+2 << "/" << normIdx << " "
						<< vcount+3 << "/" << tcount+3 << "/" << normIdx << endl;
					vcount += 3;
					tcount += 3;
				}

			// If not valid DN output, just use existing facade
			} else {

				// Add material for this facade color
				mtlFile << "newmtl " << fiStr << "_bg" << endl;
				mtlFile << "Kd " << fp.bg_color.x << " "
								<< fp.bg_color.y << " "
								<< fp.bg_color.z << endl;

				// Use this material
				objFile << "usemtl " << fiStr << "_bg" << endl;

				// Write the normal
				glm::vec3 norm = glm::normalize(facadeInfo[fi].normal);
				objFile << "vn " << norm.x << " " << norm.y << " " << norm.z << endl;
				int normIdx = ++ncount;

				// Position and color buffers
				vector<glm::vec2> synthPosBuf;
				vector<glm::vec3> synthColBuf;

				// Add each triangle
				for (auto f : facadeInfo[fi].faceIDs) {
					// Write positions + background color
					glm::vec3 va = posBuf[indexBuf[3 * f + 0]];
					glm::vec3 vb = posBuf[indexBuf[3 * f + 1]];
					glm::vec3 vc = posBuf[indexBuf[3 * f + 2]];

					// Store positions in position buffer
					synthPosBuf.push_back(fp.xform * glm::vec4(va, 1.0));
					synthPosBuf.push_back(fp.xform * glm::vec4(vb, 1.0));
					synthPosBuf.push_back(fp.xform * glm::vec4(vc, 1.0));
					// Store colors in color buffer
					synthColBuf.push_back(fp.bg_color);
					synthColBuf.push_back(fp.bg_color);
					synthColBuf.push_back(fp.bg_color);

					objFile << "v " << va.x << " " << va.y << " " << va.z << " "
						<< fp.bg_color.x << " " << fp.bg_color.y << " " << fp.bg_color.z << endl;
					objFile << "v " << vb.x << " " << vb.y << " " << vb.z << " "
						<< fp.bg_color.x << " " << fp.bg_color.y << " " << fp.bg_color.z << endl;
					objFile << "v " << vc.x << " " << vc.y << " " << vc.z << " "
						<< fp.bg_color.x << " " << fp.bg_color.y << " " << fp.bg_color.z << endl;

					// Write indices
					objFile << "f " << vcount+1 << "//" << normIdx << " "
						<< vcount+2 << "//" << normIdx << " "
						<< vcount+3 << "//" << normIdx << endl;
					vcount += 3;
				}

				// Upload positions and colors
				glBindBuffer(GL_ARRAY_BUFFER, pbuf);
				glBufferData(GL_ARRAY_BUFFER, synthPosBuf.size() * sizeof(synthPosBuf[0]),
					synthPosBuf.data(), GL_STATIC_DRAW);
				glBindBuffer(GL_ARRAY_BUFFER, cbuf);
				glBufferData(GL_ARRAY_BUFFER, synthColBuf.size() * sizeof(synthColBuf[0]),
					synthColBuf.data(), GL_STATIC_DRAW);

				// Upload xform matrix
				glm::mat4 glXform(1.0);
				glXform[0][0] = 2.0 / fp.geom_size.x;
				glXform[1][1] = 2.0 / fp.geom_size.y;
				glXform[3][0] = -1.0;
				glXform[3][1] = -1.0;
				glUniformMatrix4fv(xformLoc, 1, GL_FALSE, glm::value_ptr(glXform));

				// Resize framebuffer texture
				glm::uvec2 synthSizePx = facadeInfo[fi].size;
				glBindTexture(GL_TEXTURE_2D, fbtex);
				glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, synthSizePx.x, synthSizePx.y, 0,
					GL_RGBA, GL_UNSIGNED_BYTE, NULL);
				glBindTexture(GL_TEXTURE_2D, 0);
				glViewport(0, 0, synthSizePx.x, synthSizePx.y);
				glClear(GL_COLOR_BUFFER_BIT);

				// Draw facade geometry onto texture
				glDrawArrays(GL_TRIANGLES, 0, synthPosBuf.size());

				// Download synth texture
				cv::Mat synthImage(synthSizePx.y, synthSizePx.x, CV_8UC4);
				glBindTexture(GL_TEXTURE_2D, fbtex);
				glGetTexImage(GL_TEXTURE_2D, 0, GL_BGRA, GL_UNSIGNED_BYTE, synthImage.data);
				cv::flip(synthImage, synthImage, 0);

				// Save to disk
				fs::path synthPath = synthDir / (cluster + "_" + fiStr + ".png");
				cv::imwrite(synthPath.string(), synthImage);
			}

		// If no DN metadata, just use existing facade
		} else {
			// Load one view of this facade (pick the first one we have)
			fs::path facadePath = modelDir / "facade" / fiStr /
				(*(facadeInfo[fi].inSats.begin()) + "_ps.png");
			cv::Mat facadeImage = cv::imread(facadePath.string(), CV_LOAD_IMAGE_UNCHANGED);
			if (!facadeImage.data) {
				cout << "Facade " << fiStr << " texture " << facadePath.filename()
					<< " missing!" << endl;
				continue;
			}

			// Separate alpha channel
			cv::Mat aMask(facadeImage.size(), CV_8UC1);
			cv::mixChannels(vector<cv::Mat>{ facadeImage }, vector<cv::Mat>{ aMask }, { 3, 0 });

			// Get mean color
			cv::Scalar meanCol = cv::mean(facadeImage, aMask);
			glm::vec3 drawCol(meanCol[2] / 255.0, meanCol[1] / 255.0, meanCol[0] / 255.0);

			// Add material for this facade color
			mtlFile << "newmtl " << fiStr << "_bg" << endl;
			mtlFile << "Kd " << drawCol.x << " " << drawCol.y << " " << drawCol.z << endl;

			// Use texture-mapped material
			objFile << "usemtl " << fiStr << "_bg" << endl;

			// Write the normal
			glm::vec3 norm = glm::normalize(facadeInfo[fi].normal);
			objFile << "vn " << norm.x << " " << norm.y << " " << norm.z << endl;
			int normIdx = ++ncount;

			// Add each triangle
			for (auto f : facadeInfo[fi].faceIDs) {
				// Write positions
				glm::vec3 va = posBuf[indexBuf[3 * f + 0]];
				glm::vec3 vb = posBuf[indexBuf[3 * f + 1]];
				glm::vec3 vc = posBuf[indexBuf[3 * f + 2]];
				objFile << "v " << va.x << " " << va.y << " " << va.z << " "
					<< drawCol.x << " " << drawCol.y << " " << drawCol.z << endl;
				objFile << "v " << vb.x << " " << vb.y << " " << vb.z << " "
					<< drawCol.x << " " << drawCol.y << " " << drawCol.z << endl;
				objFile << "v " << vc.x << " " << vc.y << " " << vc.z << " "
					<< drawCol.x << " " << drawCol.y << " " << drawCol.z << endl;

				// Write indices
				objFile << "f " << vcount+1 << "//" << normIdx << " "
					<< vcount+2 << "//" << normIdx << " "
					<< vcount+3 << "//" << normIdx << endl;
				vcount += 3;
			}
		}
	}


	// Resize framebuffer
	glBindTexture(GL_TEXTURE_2D, fbtex);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, atlasSize.x, atlasSize.y, 0,
		GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	glBindTexture(GL_TEXTURE_2D, 0);
	glViewport(0, 0, atlasSize.x, atlasSize.y);

	// Upload atlas texture coords
	glBindBuffer(GL_ARRAY_BUFFER, pbuf);
	glBufferData(GL_ARRAY_BUFFER, atlasTCBuf.size() * sizeof(atlasTCBuf[0]),
		atlasTCBuf.data(), GL_STATIC_DRAW);
	// Create index buffer
	GLuint ibuf = ctx.genBuffer();
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibuf);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, indexBuf.size() * sizeof(indexBuf[0]),
		indexBuf.data(), GL_STATIC_DRAW);

	// Create vertex array object
	GLuint vao2 = ctx.genVAO();
	glBindVertexArray(vao2);
	glBindBuffer(GL_ARRAY_BUFFER, pbuf);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(glm::vec2), 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibuf);

	// Vertex shader
	static const string vsh_str2 = R"(
		#version 460

		layout(location = 0) in vec2 tc;

		smooth out vec2 fragTC;

		uniform mat4 tcXform;

		void main() {
			gl_Position = vec4(tc * vec2(2) - vec2(1), 0.0, 1.0);
			fragTC = vec2(tcXform * vec4(tc, 0.0, 1.0));
		})";
	// Fragment shader
	static const string fsh_str2 = R"(
		#version 460

		smooth in vec2 fragTC;

		out vec4 outCol;

		uniform sampler2D texSampler;

		void main() {
			outCol = texture(texSampler, fragTC);
		})";

	// Compile and link shaders
	shaders.clear();
	shaders.push_back(ctx.compileShader(GL_VERTEX_SHADER, vsh_str2));
	shaders.push_back(ctx.compileShader(GL_FRAGMENT_SHADER, fsh_str2));
	GLuint program2 = ctx.linkProgram(shaders);
	GLuint xformLoc2 = glGetUniformLocation(program2, "tcXform");
	glUseProgram(program2);

	// Create facade texture
	GLuint facadeTex = ctx.genTexture();
	glBindTexture(GL_TEXTURE_2D, facadeTex);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);


	// Output synthetic texture atlas
	glClear(GL_COLOR_BUFFER_BIT);

	// Iterate over all facades
	for (size_t fi = 0; fi < facadeInfo.size(); fi++) {
		string fiStr; {
			stringstream ss;
			ss << setw(4) << setfill('0') << fi;
			fiStr = ss.str();
		}

		if (facadeParams.count(fi)) {
			const FacadeParams& fp = facadeParams[fi];

			// Load facade image
			cv::Mat facadeImage;
			if (fp.roof) {
				fs::path facadePath = outDir / "image" / (cluster + "_" + fiStr + ".png");
				facadeImage = cv::imread(facadePath.string(), CV_LOAD_IMAGE_UNCHANGED);
			} else {
				fs::path facadePath = outDir / "synth" / (cluster + "_" + fiStr + ".png");
				facadeImage = cv::imread(facadePath.string(), CV_LOAD_IMAGE_UNCHANGED);
			}

			// Upload facade texture to OpenGL
			cv::flip(facadeImage, facadeImage, 0);
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, facadeImage.cols, facadeImage.rows, 0,
				GL_BGRA, GL_UNSIGNED_BYTE, facadeImage.data);

			// Set texture coordinate transformation matrix
			cv::Rect2f atlasBB = facadeInfo[fi].atlasBB;
			glm::mat4 xlate(1.0);
			xlate[3] = glm::vec4(-atlasBB.x, -atlasBB.y, 0.0, 1.0);
			glm::mat4 scale(1.0);
			scale[0][0] = 1.0 / atlasBB.width;
			scale[1][1] = 1.0 / atlasBB.height;
			glm::mat4 glXform = scale * xlate;
			glUniformMatrix4fv(xformLoc2, 1, GL_FALSE, glm::value_ptr(glXform));

			// Draw each face in this facade group
			for (auto f : facadeInfo[fi].faceIDs)
				glDrawElements(GL_TRIANGLES, 3, GL_UNSIGNED_INT, (GLvoid*)(3 * f * sizeof(glm::uint)));
		}
	}

	// Download atlas texture
	cv::Mat atlasImage(atlasSize.y, atlasSize.x, CV_8UC4);
	glBindTexture(GL_TEXTURE_2D, fbtex);
	glGetTexImage(GL_TEXTURE_2D, 0, GL_BGRA, GL_UNSIGNED_BYTE, atlasImage.data);
	cv::flip(atlasImage, atlasImage, 0);

	// Save to disk
	fs::path synthAtlasPath = outDir / (cluster + "_synth_atlas.png");
	cv::imwrite(synthAtlasPath.string(), atlasImage);


	// Output best scores texture atlas
	glBindTexture(GL_TEXTURE_2D, facadeTex);
	glClear(GL_COLOR_BUFFER_BIT);

	// Iterate over all facades
	for (size_t fi = 0; fi < facadeInfo.size(); fi++) {
		string fiStr; {
			stringstream ss;
			ss << setw(4) << setfill('0') << fi;
			fiStr = ss.str();
		}

		if (facadeParams.count(fi)) {
			const FacadeParams& fp = facadeParams[fi];

			// Load facade image
			cv::Mat facadeImage;
			fs::path facadePath = outDir / "image" / (cluster + "_" + fiStr + ".png");
			facadeImage = cv::imread(facadePath.string(), CV_LOAD_IMAGE_UNCHANGED);

			// Upload facade texture to OpenGL
			cv::flip(facadeImage, facadeImage, 0);
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, facadeImage.cols, facadeImage.rows, 0,
				GL_BGRA, GL_UNSIGNED_BYTE, facadeImage.data);

			// Set texture coordinate transformation matrix
			cv::Rect2f atlasBB = facadeInfo[fi].atlasBB;
			glm::mat4 xlate(1.0);
			xlate[3] = glm::vec4(-atlasBB.x, -atlasBB.y, 0.0, 1.0);
			glm::mat4 scale(1.0);
			scale[0][0] = 1.0 / atlasBB.width;
			scale[1][1] = 1.0 / atlasBB.height;
			glm::mat4 glXform = scale * xlate;
			glUniformMatrix4fv(xformLoc2, 1, GL_FALSE, glm::value_ptr(glXform));

			// Draw each face in this facade group
			for (auto f : facadeInfo[fi].faceIDs)
				glDrawElements(GL_TRIANGLES, 3, GL_UNSIGNED_INT, (GLvoid*)(3 * f * sizeof(glm::uint)));
		}
	}

	// Download atlas texture
	glBindTexture(GL_TEXTURE_2D, fbtex);
	glGetTexImage(GL_TEXTURE_2D, 0, GL_BGRA, GL_UNSIGNED_BYTE, atlasImage.data);
	cv::flip(atlasImage, atlasImage, 0);

	// Save to disk
	fs::path bestScoresAtlasPath = outDir / (cluster + "_bestscores_atlas.png");
	cv::imwrite(bestScoresAtlasPath.string(), atlasImage);


	// Output histeq best scores texture atlas
	glBindTexture(GL_TEXTURE_2D, facadeTex);
	glClear(GL_COLOR_BUFFER_BIT);

	// Iterate over all facades
	for (size_t fi = 0; fi < facadeInfo.size(); fi++) {
		string fiStr; {
			stringstream ss;
			ss << setw(4) << setfill('0') << fi;
			fiStr = ss.str();
		}

		if (facadeParams.count(fi)) {
			const FacadeParams& fp = facadeParams[fi];

			// Load facade image
			cv::Mat facadeImage;
			fs::path facadePath = outDir / "image-histeq" / (cluster + "_" + fiStr + ".png");
			facadeImage = cv::imread(facadePath.string(), CV_LOAD_IMAGE_UNCHANGED);

			// Upload facade texture to OpenGL
			cv::flip(facadeImage, facadeImage, 0);
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, facadeImage.cols, facadeImage.rows, 0,
				GL_BGRA, GL_UNSIGNED_BYTE, facadeImage.data);

			// Set texture coordinate transformation matrix
			cv::Rect2f atlasBB = facadeInfo[fi].atlasBB;
			glm::mat4 xlate(1.0);
			xlate[3] = glm::vec4(-atlasBB.x, -atlasBB.y, 0.0, 1.0);
			glm::mat4 scale(1.0);
			scale[0][0] = 1.0 / atlasBB.width;
			scale[1][1] = 1.0 / atlasBB.height;
			glm::mat4 glXform = scale * xlate;
			glUniformMatrix4fv(xformLoc2, 1, GL_FALSE, glm::value_ptr(glXform));

			// Draw each face in this facade group
			for (auto f : facadeInfo[fi].faceIDs)
				glDrawElements(GL_TRIANGLES, 3, GL_UNSIGNED_INT, (GLvoid*)(3 * f * sizeof(glm::uint)));
		}
	}

	// Download atlas texture
	glBindTexture(GL_TEXTURE_2D, fbtex);
	glGetTexImage(GL_TEXTURE_2D, 0, GL_BGRA, GL_UNSIGNED_BYTE, atlasImage.data);
	cv::flip(atlasImage, atlasImage, 0);

	// Save to disk
	fs::path histeqAtlasPath = outDir / (cluster + "_histeq_atlas.png");
	cv::imwrite(histeqAtlasPath.string(), atlasImage);


	// Output textured synth model
	fs::path objPath2 = outDir / (cluster + "_synth_texture.obj");
	fs::path mtlPath2 = objPath2; mtlPath2.replace_extension(".mtl");
	ofstream objFile2(objPath2);
	ofstream mtlFile2(mtlPath2);

	// Create atlas materials
	mtlFile2 << "newmtl synth_atlas" << endl;
	mtlFile2 << "map_Kd " << synthAtlasPath.filename().string() << endl;
	mtlFile2 << "newmtl bestscores_atlas" << endl;
	mtlFile2 << "map_Kd " << bestScoresAtlasPath.filename().string() << endl;
	mtlFile2 << "newmtl histeq_atlas" << endl;
	mtlFile2 << "map_Kd " << histeqAtlasPath.filename().string() << endl;

	// Use synth material
	objFile2 << "mtllib " << mtlPath2.filename().string() << endl;
	objFile2 << "usemtl synth_atlas" << endl;
	objFile2 << endl;

	// Write all vertex positions
	for (auto v : posBuf)
		objFile2 << "v " << v.x << " " << v.y << " " << v.z << endl;
	objFile2 << endl;
	// Write all vertex normals
	for (auto n : normBuf)
		objFile2 << "vn " << n.x << " " << n.y << " " << n.z << endl;
	objFile2 << endl;
	// Write all texture coordinates
	for (auto t : atlasTCBuf)
		objFile2 << "vt " << t.x << " " << t.y << endl;
	objFile2 << endl;

	// Write all faces
	for (size_t f = 0; f < indexBuf.size(); f += 3) {
		objFile2 << "f ";
		objFile2 << indexBuf[f + 0] + 1 << "/"
				 << indexBuf[f + 0] + 1 << "/"
				 << indexBuf[f + 0] + 1 << " ";
		objFile2 << indexBuf[f + 1] + 1 << "/"
				 << indexBuf[f + 1] + 1 << "/"
				 << indexBuf[f + 1] + 1 << " ";
		objFile2 << indexBuf[f + 2] + 1 << "/"
				 << indexBuf[f + 2] + 1 << "/"
				 << indexBuf[f + 2] + 1 << endl;
	}
}

// Combine .obj outputs of specified clusters, using synthesized texture atlases
void Building::combineOutput(fs::path outputDir, string region, string model,
	vector<Building>& bldgs) {

	// Get bounding box of all buildings
	glm::vec3 minBB(FLT_MAX), maxBB(-FLT_MAX);
	for (auto& b : bldgs) {
		minBB = glm::min(minBB, b.minBB + b.origin);
		maxBB = glm::max(maxBB, b.maxBB + b.origin);
	}
	glm::vec3 center = glm::vec3(glm::vec2(minBB + (maxBB - minBB) / 2.0f), 0.0f);

	// Create output directory
	fs::path outDir = outputDir / region / model / "all";
	if (fs::exists(outDir))
		fs::remove_all(outDir);
	fs::create_directories(outDir);

	// Create output OBJ and MTL files
	fs::path objPath = outDir / (region + "_all_synth_texture.obj");
	fs::path mtlPath = objPath; mtlPath.replace_extension(".mtl");
	ofstream objFile(objPath);
	ofstream mtlFile(mtlPath);

	objFile << "mtllib " << mtlPath.filename().string() << endl << endl;

	// Iterate over all buildings
	int idx_offset = 0;
	for (auto& b : bldgs) {
		// Copy synth atlas
		string atlasName = b.cluster + "_synth_atlas.png";
		fs::path atlasPath = outputDir / region / model / b.cluster / atlasName;
		if (!fs::exists(atlasPath)) {
			cout << atlasPath << " not found! Skipping..." << endl;
			continue;
		}
		fs::copy_file(outputDir / region / model / b.cluster / atlasName,
			outDir / atlasName);

		// Add material
		mtlFile << "newmtl " << b.cluster << "_atlas" << endl;
		mtlFile << "map_Kd " << atlasName << endl;

		// Add cluster group, use new material
		objFile << "g " << b.cluster << endl;
		objFile << "usemtl " << b.cluster << "_atlas" << endl;
		objFile << endl;

		// Write all vertex positions
		for (auto v : b.posBuf) {
			v = v + b.origin - center;
			objFile << "v " << v.x << " " << v.y << " " << v.z << endl;
		}
		objFile << endl;
		// Write all vertex normals
		for (auto n : b.normBuf)
			objFile << "vn " << n.x << " " << n.y << " " << n.z << endl;
		objFile << endl;
		// Write all texture coords
		for (auto t : b.atlasTCBuf)
			objFile << "vt " << t.x << " " << t.y << endl;
		objFile << endl;

		// Write all faces
		for (size_t f = 0; f < b.indexBuf.size(); f += 3) {
			objFile << "f ";
			objFile << b.indexBuf[f + 0] + idx_offset + 1 << "/"
					<< b.indexBuf[f + 0] + idx_offset + 1 << "/"
					<< b.indexBuf[f + 0] + idx_offset + 1 << " ";
			objFile << b.indexBuf[f + 1] + idx_offset + 1 << "/"
					<< b.indexBuf[f + 1] + idx_offset + 1 << "/"
					<< b.indexBuf[f + 1] + idx_offset + 1 << " ";
			objFile << b.indexBuf[f + 2] + idx_offset + 1 << "/"
					<< b.indexBuf[f + 2] + idx_offset + 1 << "/"
					<< b.indexBuf[f + 2] + idx_offset + 1 << endl;
		}
		objFile << endl;

		idx_offset += b.posBuf.size();
	}
}

// Create cluster masks from satellite images and building geometry
void Building::createClusterMasks(fs::path dataDir, map<string, Satellite>& sats,
	string region, string model, vector<Building>& bldgs) {

	// Create cluster mask directory
	fs::path clusterMaskDir = dataDir / "clusterMasks" / region / model;
	if (fs::exists(clusterMaskDir))
		fs::remove_all(clusterMaskDir);
	fs::create_directories(clusterMaskDir);

	// Iterate over all satellite images
	for (auto& si : sats) {
		cout << si.first << endl;

		map<int, float> bldgBase;	// Lowest point along satellite projUp vector, per bldg
		vector<int> bldgOrder;		// Order to draw bldgs
		// Compute drawing order of buildings
		for (int bi = 0; bi < bldgs.size(); bi++) {
			Building& b = bldgs[bi];
			// Skip building if not seen by this satellite
			if (!b.satInfo.count(si.first)) continue;

			// Calculate lowest point along projected up vector
			float minUp = FLT_MAX;
			for (auto v : b.satTCBufs[si.first]) {
				if (v.x < -0.5 || v.y < -0.5) continue;

				// Convert from UV to pixels
				v = glm::vec2(util::SpatXform::uv2px(glm::vec3(v, 0.0), b.satInfo[si.first].roi));
				v.x = v.x + b.satInfo[si.first].roi.x;
				v.y = v.y + b.satInfo[si.first].roi.y;
				// Calculate distance along projected up vector
				float upLen = glm::dot(si.second.projUp, v);
				if (upLen < minUp) minUp = upLen;
			}
			bldgBase[bi] = minUp;
			bldgOrder.push_back(bi);
		}
		// Sort clusters in descending base order
		sort(bldgOrder.begin(), bldgOrder.end(), [&](const int& a, const int& b) -> bool {
			return bldgBase[a] > bldgBase[b];
		});

		cv::Mat clusterMask = cv::Mat::zeros(si.second.satImg.size(), CV_8UC1);
		// Draw each building onto the cluster mask
		for (int bi : bldgOrder) {
			Building& b = bldgs[bi];
			// Skip building if not seen by this satellite
			if (!b.satInfo.count(si.first)) continue;

			// Draw all triangles in the building
			for (int f = 0; f < b.indexBuf.size(); f += 3) {
				// Skip if any vert is invalid
				if (b.satTCBufs[si.first][b.indexBuf[f + 0]].x < -0.5 ||
					b.satTCBufs[si.first][b.indexBuf[f + 0]].y < -0.5 ||
					b.satTCBufs[si.first][b.indexBuf[f + 1]].x < -0.5 ||
					b.satTCBufs[si.first][b.indexBuf[f + 1]].y < -0.5 ||
					b.satTCBufs[si.first][b.indexBuf[f + 1]].x < -0.5 ||
					b.satTCBufs[si.first][b.indexBuf[f + 1]].y < -0.5)
					continue;

				// Get triangle verts in pixels
				vector<cv::Point> pts;
				for (int fi = 0; fi < 3; fi++) {
					glm::vec2 v = b.satTCBufs[si.first][b.indexBuf[f + fi]];
					v = glm::vec2(util::SpatXform::uv2px(glm::vec3(v, 0.0), b.satInfo[si.first].roi));
					v.x = v.x + b.satInfo[si.first].roi.x;
					v.y = v.y + b.satInfo[si.first].roi.y;
					pts.push_back(cv::Point(v.x, v.y));
				}

				// Draw the triangle
				cv::fillConvexPoly(clusterMask, pts, cv::Scalar::all(stoi(b.cluster)));
			}
		}

		// Save the cluster mask
		fs::path clusterMaskPath = clusterMaskDir / (si.first + "_cid.png");
		cv::imwrite(clusterMaskPath.string(), clusterMask);
	}

	// Get cropped and facade versions for each building
	for (auto& b : bldgs) {
		cout << b.cluster << endl;
		b.genClusterMasks(dataDir);
	}
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
	map<string, int> satVertCount;
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
		satVertCount[sat.name] = 0;
	}

	// Add faces to geometry buffers
	for (size_t s = 0; s < shapes.size(); s++) {
		size_t idx_offset = 0;
		for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
			int fv = shapes[s].mesh.num_face_vertices[f];
			map<string, vector<glm::vec2>> satPts;

			// Skip degenerate and down-facing triangles
			glm::vec3 normal;
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
//				cout << "Degenerate triangle! "
//					<< lengths[0] << " " << lengths[1] << " " << lengths[2] << endl;
				continue;
			}

			// Skip if facing downward
			normal = glm::normalize(glm::cross(vb - va, vc - va));
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

				// Skip if entirely outside of roi
				if ((projPts[0].x < -0.5 || projPts[0].x > 1.5 ||
					projPts[0].y < -0.5 || projPts[0].y > 1.5) ||
					(projPts[1].x < -0.5 || projPts[1].x > 1.5 ||
					projPts[1].y < -0.5 || projPts[1].y > 1.5) ||
					(projPts[2].x < -0.5 || projPts[2].x > 1.5 ||
					projPts[2].y < -0.5 || projPts[2].y > 1.5)) continue;

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

				// Add position
				posBuf.push_back({
					attrib.vertices[3 * idx.vertex_index + 0],
					attrib.vertices[3 * idx.vertex_index + 1],
					attrib.vertices[3 * idx.vertex_index + 2]});

				// Use computed normal if mesh doesn't have normals
				if (idx.normal_index < 0)
					normBuf.push_back(normal);
				// Otherwise use mesh normals
				else
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
					else {
						satTCBufs[satName].push_back(satPts[satName][v]);
						satVertCount[satName]++;
					}
				}

				// Update bounding box
				minBB = glm::min(minBB, posBuf.back());
				maxBB = glm::max(maxBB, posBuf.back());
			}

			idx_offset += fv;
		}
	}

	// Remove any unused satellites
	for (auto si : satVertCount) {
		if (si.second == 0) {
			satInfo.erase(si.first);
			satTCBufs.erase(si.first);
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
				if (ta.x >= -0.5 && ta.y >= -0.5 &&
					tb.x >= -0.5 && tb.y >= -0.5 &&
					tc.x >= -0.5 && tc.y >= -0.5) {
					observed = true;
					break;
				}
			}
			if (!observed) continue;
			inSats.insert(si.first);
		}
		// Skip facades without any observing satellites
		if (inSats.empty()) continue;

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
	modelDir = dataDir / "regions" / region / model / cluster;
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
			if (ta.x < -0.5 || ta.y < -0.5 ||
				tb.x < -0.5 || tb.y < -0.5 ||
				tc.x < -0.5 || tc.y < -0.5) continue;

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

// Save cropped versions of all satellite images
void Building::genTextures(fs::path dataDir, map<string, Satellite>& sats) {
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
			if (geoTC[0].x < -0.5 || geoTC[0].y < -0.5 ||
				geoTC[1].x < -0.5 || geoTC[1].y < -0.5 ||
				geoTC[2].x < -0.5 || geoTC[2].y < -0.5) return;

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

	// Upload satellite images to textures
	map<string, GLuint> satTexs;
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
	}
}

// Save cropped and facade versions of cluster masks
void Building::genClusterMasks(fs::path dataDir) {
	// Check for existence of cluster masks
	fs::path clusterMaskDir = dataDir / "clusterMasks" / region / model;
	if (!fs::exists(clusterMaskDir)) return;

	// Make sure out sat dir exists
	fs::path satDir = modelDir / "sat";
	if (!fs::exists(satDir))
		fs::create_directory(satDir);

	// Iterate over all used satellites
	for (auto& si : satInfo) {
		// Load the cluster mask
		fs::path clusterMaskPath = clusterMaskDir / (si.first + "_cid.png");
		cv::Mat clusterMask = cv::imread(clusterMaskPath.string(), CV_LOAD_IMAGE_UNCHANGED);
		if (!clusterMask.data) {
			cout << "Couldn't read cluster mask " << clusterMaskPath << endl;
			continue;
		}

		// Save the cropped cluster mask
		fs::path clusterMaskOutPath = satDir / clusterMaskPath.filename();
		cv::imwrite(clusterMaskOutPath.string(), clusterMask(si.second.roi));
	}


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
			if (geoTC[0].x < -0.5 || geoTC[0].y < -0.5 ||
				geoTC[1].x < -0.5 || geoTC[1].y < -0.5 ||
				geoTC[2].x < -0.5 || geoTC[2].y < -0.5) return;

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

	// Upload cluster masks as textures
	map<string, GLuint> maskTexs;
	for (auto& si : satInfo) {
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

	// Make sure facade dir exists
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
			// Bind mask texture
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
