#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <experimental/filesystem>
#include "json.hpp"
using namespace std;
using json = nlohmann::json;
namespace fs = std::experimental::filesystem;

// Global state
fs::path regionDir;
fs::path satelliteDir;
fs::path dataDir;
fs::path outputDir;

// Functions
void genConfig(fs::path configPath);
void readConfig(fs::path configPath);

int main(int argc, char** argv) {
	try {
		// Parse commandline parameters

		// Check if config file exists
		fs::path configPath = "config.json";
		if (!fs::exists(configPath)) {
			cout << "Configuration file " << configPath << " does not exist!" << endl;
			// Ask to generate a template file
			while (true) {
				cout << "Generate template? (y/n): ";
				char c;
				cin >> c;
				if (c == 'y' || c == 'Y') {
					// Generate a template config file
					genConfig(configPath);
					break;
				} else if (c == 'n' || c == 'N') {
					break;
				}
			}
			return 0;
		}
		// Read config file
		readConfig(configPath);

	// Handle any exceptions
	} catch (const exception& e) {
		cerr << e.what() << endl;
		return -1;
	}

	return 0;
}

// Generates a template config file at the specified path
void genConfig(fs::path configPath) {
	try {
		json templateConfig;
		// Directory settings
		templateConfig["regionDir"] = "";
		templateConfig["satelliteDir"] = "";
		templateConfig["dataDir"] = "";
		templateConfig["outputDir"] = "";

		// Write template config to file
		ofstream templateFile;
		templateFile.exceptions(ios::badbit | ios::failbit | ios::eofbit);
		templateFile.open(configPath);
		templateFile << setw(4) << templateConfig;

	} catch (const exception& e) {
		stringstream ss;
		ss << "Failed to generate template config file: " << e.what();
		throw runtime_error(ss.str());
	}
}

// Loads the configuration file and stores its parameters into global variables
void readConfig(fs::path configPath) {
	try {
		// Read the config file
		ifstream configFile(configPath);
		json config;
		configFile >> config;

		// Store parameters
		regionDir = fs::path(config.at("regionDir"));
		satelliteDir = fs::path(config.at("satelliteDir"));
		dataDir = fs::path(config.at("dataDir"));
		outputDir = fs::path(config.at("outputDir"));

	// Add message to errors while reading
	} catch (const exception& e) {
		stringstream ss;
		ss << "Failed to read config file: " << e.what();
		throw runtime_error(ss.str());
	}

	// Check for existence of input directories
	if (!fs::exists(regionDir)) {
		stringstream ss;
		ss << "Region directory " << regionDir << " does not exist!";
		throw runtime_error(ss.str());
	}
	if (!fs::is_directory(regionDir)) {
		stringstream ss;
		ss << "Region path " << regionDir << " not a directory!";
		throw runtime_error(ss.str());
	}
	if (!fs::exists(satelliteDir)) {
		stringstream ss;
		ss << "Satellite directory " << satelliteDir << " does not exist!";
		throw runtime_error(ss.str());
	}
	if (!fs::is_directory(satelliteDir)) {
		stringstream ss;
		ss << "Satellite path " << satelliteDir << " not a directory!";
		throw runtime_error(ss.str());
	}
}
