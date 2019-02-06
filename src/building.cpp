#include "building.hpp"
using namespace std;
namespace fs = std::experimental::filesystem;

// Generate building data from input directory, and save it to data directory
void Building::generate(fs::path inputClusterDir, fs::path dataClusterDir,
	string model, vector<Satellite>& sats) {
}

// Load building data from data directory
void Building::load(fs::path dataClusterDir, string model) {
}
