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
	string region;			// Which region to process
	vector<int> clusters;	// Which clusters to process (empty -> all in region)
	string model;			// Which model to use (e.g., cgv_r, cgv_a, etc.)
	bool quit;				// Do nothing (used for --help)
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

Options::Options() : model("cgv_r"), quit(false), configPath("config.json") {}

// Parses commandline arguments and returns a set of options
Options parseCmd(int argc, char** argv) {
	auto usage = [&](ostream& out) -> void {
		out << "Usage: " << argv[0] << " region [cluster IDs] [OPTIONS]" << endl;
		out << "Options:" << endl;
		out << "    --config <path>      Specify config file to use" << endl;
		out << "    -h, --help           Display this help and exit" << endl;
		out << "    -m, --model <name>   Which model to process" << endl;
		out << "                           Defaults to cgv_r" << endl;
	};

	Options opts;

	try {
		// Get region
		if (argc < 2)
			throw runtime_error("Too few arguments! Expected region");
		opts.region = argv[1];
		// Check for --help
		if (opts.region == "-h" || opts.region == "--help") {
			usage(cout);
			opts.quit = true;
			return opts;
		}

		// Get cluster IDs
		int a;
		for (a = 2; a < argc; a++) {
			// Try to get an integer from the arg string
			try {
				string arg = argv[a];
				opts.clusters.push_back(stoi(arg));
			// If failed, stop trying to get cluster IDs
			} catch (const exception& e) {
				break;
			}
		}

		// Look at all remaining arguments
		for (; a < argc; a++) {
			string arg(argv[a]);
			// Specify a config file to use
			if (arg == "--config") {
				if (a+1 < argc)
					opts.configPath = fs::path(argv[++a]);
				else
					throw runtime_error("Expected <path> after " + arg);

			// Ask for help
			} else if (arg == "-h" || arg == "--help") {
				usage(cout);
				opts.quit = true;
				break;

			// Specify model to process
			} else if (arg == "-m" || arg == "--model") {
				if (a+1 < argc)
					opts.model = argv[++a];
				else
					throw runtime_error("Expected <name> after " + arg);

			// Unknown argument
			} else {
				stringstream ss;
				ss << "Unknown argument " << arg;
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
