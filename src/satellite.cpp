#include "satellite.hpp"
#include <experimental/filesystem>
using namespace std;
namespace fs = std::experimental::filesystem;

int cvType(GDALDataType gdt) {
	switch (gdt) {
	case GDT_Byte: return CV_8UC1;
	case GDT_UInt16: return CV_16UC1;
	case GDT_Int16: return CV_16SC1;
	case GDT_Int32: return CV_32SC1;
	case GDT_Float32: return CV_32FC1;
	case GDT_Float64: return CV_64FC1;
	default: return -1;
	}
};

// Global GDAL initialization
class GDALInit {
public:
	GDALInit() { GDALAllRegister(); }
};
static GDALInit gdalInit;

Satellite::Satellite(string filename) {
	// Get the satellite name from the filename
	fs::path p(filename);
	name = p.stem().string();

	// Open the dataset
	GDALDataset* satFile = (GDALDataset*)GDALOpen(filename.c_str(), GA_ReadOnly);
	if (!satFile)
		throw runtime_error("Error opening satellite dataset");

	// Extract RPC info
	int ret = GDALExtractRPCInfo(satFile->GetMetadata("RPC"), &rpcInfo);
	if (!ret)
		throw runtime_error("Error extracting RPC info");

	// Create transformer object
	rpcXformer = GDALCreateRPCTransformer(&rpcInfo, TRUE, 0.0, NULL);
	if (!rpcXformer)
		throw runtime_error("Error creating RPC transformer");

	// Get each band as an OpenCV Mat
	vector<cv::Mat> bands;
	for (int i = 1; i <= satFile->GetRasterCount(); i++) {
		GDALRasterBand* pband = satFile->GetRasterBand(i);
		cv::Mat ch = cv::Mat::zeros(pband->GetYSize(), pband->GetXSize(), cvType(pband->GetRasterDataType()));
		CPLErr err = pband->RasterIO(GF_Read, 0, 0, pband->GetXSize(), pband->GetYSize(),
			ch.data, ch.cols, ch.rows, pband->GetRasterDataType(), 0, 0, NULL);
		if (err != CE_None)
			throw runtime_error("Error reading raster band");
		ch.convertTo(ch, CV_8U, 255);
		bands.push_back(ch);
	}
	// Swap R and B channels for OpenCV
	if (bands.size() == 3)
		swap(bands[0], bands[2]);
	// Merge all channels together
	cv::merge(bands, satImg);

	// Close the dataset
	GDALClose(satFile);
	satFile = NULL;

	// Get lat/long at image center
	glm::dvec3 pt(satImg.cols / 2, satImg.rows / 2, 0.0);
	int succ;
	GDALRPCTransform(rpcXformer, TRUE, 1, &pt.x, &pt.y, &pt.z, &succ);
	// Get coordinates of the same point at higher altitude
	pt.z = 100.0;
	GDALRPCTransform(rpcXformer, FALSE, 1, &pt.x, &pt.y, &pt.z, &succ);
	// Normalize the projected up vector
	projUp = glm::normalize(glm::vec2(pt) - glm::vec2(satImg.cols / 2, satImg.rows / 2));
}

Satellite::~Satellite() {
	if (rpcXformer) {
		GDALDestroyRPCTransformer(rpcXformer);
		rpcXformer = NULL;
	}
}

// Move constructor
Satellite::Satellite(Satellite&& other) {
	satImg = other.satImg;
	rpcInfo = other.rpcInfo;
	rpcXformer = other.rpcXformer;
	bb = other.bb;
	name = other.name;
	projUp = other.projUp;

	other.rpcXformer = NULL;
}

// Calculate the bounding box based on a set of projected points
void Satellite::calcBB(vector<cv::Point2f> allPts, int border) {
	bb = cv::boundingRect(allPts);
	// Expand bounding rect on all sides
	bb -= cv::Point(border, border);
	bb += cv::Size(2 * border, 2 * border);
	// Make sure it's still within the image bounds
	bb &= cv::Rect(0, 0, satImg.cols, satImg.rows);
}

/*
cv::Point3d utm2px(cv::Point3d p, BuildingMetadata& bm, Satellite& sat) {

	// Add building origin
	p = p + bm.origin;

	// UTM to lat/long
	bm.utm2ll->Transform(1, &p.x, &p.y);

	// Lat/long to pixels
	int succ;
	GDALRPCTransform(sat.rpcXformer, FALSE, 1, &p.x, &p.y, &p.z, &succ);

	// Relative to bounding box
	if (!sat.bb.empty()) {
		p.x -= sat.bb.tl().x;
		p.y -= sat.bb.tl().y;
	}

	return p;
}

cv::Point3d px2utm(cv::Point3d p, BuildingMetadata& bm, Satellite& sat) {

	// Relative to bounding box
	if (!sat.bb.empty()) {
		p.x += sat.bb.tl().x;
		p.y += sat.bb.tl().y;
	}

	// Pixels to lat/long
	int succ;
	GDALRPCTransform(sat.rpcXformer, TRUE, 1, &p.x, &p.y, &p.z, &succ);

	// Lat/long to UTM
	bm.ll2utm->Transform(1, &p.x, &p.y);

	// Subtract building origin
	p = p - bm.origin;

	return p;
}
*/
