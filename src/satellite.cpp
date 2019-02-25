#include "satellite.hpp"
#include <experimental/filesystem>
using namespace std;
namespace fs = std::experimental::filesystem;

// Convert GDAL types to OpenCV types
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

Satellite::Satellite(string filename) : rpcXformer(NULL) {
	if (filename.empty()) return;

	// Get the satellite name from the filename
	fs::path p(filename);
	name = p.stem().string().substr(0, 13);

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
		// Read raster band
		GDALRasterBand* pband = satFile->GetRasterBand(i);
		GDALDataType gdt = pband->GetRasterDataType();
		cv::Mat ch = cv::Mat::zeros(pband->GetYSize(), pband->GetXSize(), cvType(gdt));
		CPLErr err = pband->RasterIO(GF_Read, 0, 0, pband->GetXSize(), pband->GetYSize(),
			ch.data, ch.cols, ch.rows, gdt, 0, 0, NULL);
		if (err != CE_None)
			throw runtime_error("Error reading raster band");

		// Convert to 8-bit
		if (gdt == GDT_Float32 || gdt == GDT_Float64)
			ch.convertTo(ch, CV_8U, 255);
		else if (gdt != GDT_Byte)
			ch.convertTo(ch, CV_8U);

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

// Destructor
Satellite::~Satellite() {
	// Release memory
	if (rpcXformer) {
		GDALDestroyRPCTransformer(rpcXformer);
		rpcXformer = NULL;
	}
}

// Move constructor
Satellite::Satellite(Satellite&& other) {
	satImg = other.satImg;			// Does not copy image
	rpcInfo = other.rpcInfo;
	rpcXformer = other.rpcXformer;
	name = other.name;
	projUp = other.projUp;

	other.rpcXformer = NULL;
}

// Move assignment
Satellite& Satellite::operator=(Satellite&& other) {
	if (this == &other) return *this;

	// Move class members
	satImg = other.satImg;			// Does not copy image
	rpcInfo = other.rpcInfo;
	std::swap(rpcXformer, other.rpcXformer);
	name = other.name;
	projUp = other.projUp;
}

// Calculate the bounding box based on a set of projected points
cv::Rect Satellite::calcBB(vector<cv::Point2f> allPts, int border) {
	cv::Rect bb = cv::boundingRect(allPts);
	bb &= cv::Rect(0, 0, satImg.cols, satImg.rows);
	if (bb.width <= 0 || bb.height <= 0) return bb;
	// Expand bounding rect on all sides
	bb -= cv::Point(border, border);
	bb += cv::Size(2 * border, 2 * border);
	// Make sure it's still within the image bounds
	bb &= cv::Rect(0, 0, satImg.cols, satImg.rows);

	return bb;
}
