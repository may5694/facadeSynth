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

// Commandline options
struct Options {
	bool quit;				// Do nothing
	fs::path configPath;	// Path to config file

	Options();				// Set default values
};

// Functions
Options parseCmd(int argc, char** argv);
void genConfig(fs::path configPath);
void readConfig(fs::path configPath);

int main(int argc, char** argv) {
	try {
		// Parse commandline arguments
		Options opts = parseCmd(argc, argv);
		if (opts.quit) return 0;

		// Check if config file exists
		if (!fs::exists(opts.configPath)) {
			cout << "Configuration file " << opts.configPath << " does not exist!" << endl;
			// Ask to generate a template file
			while (true) {
				cout << "Generate template? (y/n): ";
				char c;
				cin >> c;
				if (c == 'y' || c == 'Y') {
					// Generate a template config file
					genConfig(opts.configPath);
					break;
				} else if (c == 'n' || c == 'N') {
					break;
				}
			}
			return 0;
		}
		// Read config file
		readConfig(opts.configPath);

	// Handle any exceptions
	} catch (const exception& e) {
		cerr << e.what() << endl;
		return -1;
	}

	return 0;
}

Options::Options() : quit(false), configPath("config.json") {}

// Parses commandline arguments and returns a set of options
Options parseCmd(int argc, char** argv) {
	auto usage = [&](ostream& out) -> void {
		out << "Usage: " << argv[0] << " [OPTIONS]" << endl;
		out << "Options:" << endl;
		out << "    --config <path>    Specify config file to use" << endl;
		out << "    -h, --help         Display this help and exit" << endl;
	};

	Options opts;

	try {
		// Look at all arguments
		for (int i = 1; i < argc; i++) {
			string arg(argv[i]);
			// Specify a config file to use
			if (arg == "--config") {
				if (i+1 < argc)
					opts.configPath = fs::path(argv[++i]);
				else
					throw runtime_error("Expected <path> after " + arg);

			// Ask for help
			} else if (arg == "-h" || arg == "--help") {
				usage(cout);
				opts.quit = true;
				break;

			// Unknown argument
			} else {
				stringstream ss;
				ss << "Unknown argument " << argv[i];
				throw runtime_error(ss.str());
			}
		}

	// Add usage to error message
	} catch (const exception& e) {
		stringstream ss;
		ss << e.what() << endl;
		usage(ss);
		throw runtime_error(ss.str());
	}

	return opts;
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
