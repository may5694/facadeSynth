#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <set>
#include <experimental/filesystem>
#include "json.hpp"
#include "satellite.hpp"
#include "building.hpp"
using namespace std;
using json = nlohmann::json;
namespace fs = std::experimental::filesystem;

// Program options
struct Options {
	// Commandline options
	string region;			// Which region to process (must be specified)
	set<int> clusters;		// Which clusters to process (empty -> all in region)
	string model;			// Which model to use (e.g., cgv_r, cgv_a, etc.)
	bool noop;				// Do nothing (used for --help)
	fs::path configPath;	// Path to config file
	bool generate;			// Generate data from input directories

	// Config file options
	fs::path regionDir;		// Dir containing regions with clusters
	fs::path satelliteDir;	// Dir containing pansharpened satellite imagery per region
	fs::path dataDir;		// Saved data output directory
	fs::path outputDir;		// Synthesized facades output directory

	// Constructor - set default values
	Options() :
		model("cgv_r"),
		noop(false),
		configPath("config.json"),
		generate(false) {}
};

// Functions
Options parseCmd(int argc, char** argv);
void genConfig(fs::path configPath);
void readConfig(Options& opts);
void checkDirectories(Options& opts);
vector<Satellite> loadSatellites(Options& opts);

// Program entry point
int main(int argc, char** argv) {
	try {
		// Parse commandline arguments
		Options opts = parseCmd(argc, argv);
		if (opts.noop) return 0;

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
		readConfig(opts);

		// Check for input and output directories
		checkDirectories(opts);

		// Load satellite images if generating cluster data
		vector<Satellite> sats;
		if (opts.generate)
			sats = loadSatellites(opts);

		// Process each cluster
		for (auto id : opts.clusters) {
			// Get cluster ID string
			string cidStr; {
				stringstream ss; ss << setw(4) << setfill('0') << id;
				cidStr = ss.str();
			}

			fs::path inputClusterIDDir = opts.regionDir / opts.region / "BuildingClusters" / cidStr;
			fs::path dataClusterIDDir = opts.dataDir / "regions" / opts.region / cidStr;
			Building b;
			// Load or generate building
			if (opts.generate) {
				cout << "Generating cluster " << cidStr << "..." << endl;
				b.generate(inputClusterIDDir, dataClusterIDDir, opts.model, sats);
			} else {
				cout << "Loading cluster " << cidStr << "..." << endl;
				b.load(dataClusterIDDir, opts.model);
			}

			// TODO: score, synthesize
		}

	// Handle any exceptions
	} catch (const exception& e) {
		cerr << e.what() << endl;
		return -1;
	}

	return 0;
}

// Parses commandline arguments and returns a set of options
Options parseCmd(int argc, char** argv) {
	auto usage = [&](ostream& out) -> void {
		out << "Usage: " << argv[0] << " region [cluster IDs] [OPTIONS]" << endl;
		out << "Options:" << endl;
		out << "    --config <path>      Specify config file to use" << endl;
		out << "    -g, --generate       Generate cluster data from input" << endl;
		out << "                           directories." << endl;
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
			opts.noop = true;
			return opts;
		}

		// Get cluster IDs
		int a;
		for (a = 2; a < argc; a++) {
			// Try to get an integer from the arg string
			try {
				string arg = argv[a];
				opts.clusters.insert(stoi(arg));
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

			// Generate cluster data from input directories
			} else if (arg == "-g" || arg == "--generate") {
				opts.generate = true;

			// Ask for help
			} else if (arg == "-h" || arg == "--help") {
				usage(cout);
				opts.noop = true;
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
void readConfig(Options& opts) {
	try {
		// Read the config file
		ifstream configFile(opts.configPath);
		json config;
		configFile >> config;

		// Store parameters
		opts.regionDir = config.at("regionDir").get<string>();
		opts.satelliteDir = config.at("satelliteDir").get<string>();
		opts.dataDir = config.at("dataDir").get<string>();
		opts.outputDir = config.at("outputDir").get<string>();

	// Add message to errors while reading
	} catch (const exception& e) {
		stringstream ss;
		ss << "Failed to read config file: " << e.what();
		throw runtime_error(ss.str());
	}
}

// Checks for input directories and creates output dirs as needed
// Also populates and checks opts.clusters
void checkDirectories(Options& opts) {
	// Look at input directories only if we're generating data
	if (opts.generate) {
		// Check for existence of input directories
		if (!fs::exists(opts.regionDir)) {
			stringstream ss;
			ss << "Region directory " << opts.regionDir << " does not exist!";
			throw runtime_error(ss.str());
		}
		if (!fs::exists(opts.satelliteDir)) {
			stringstream ss;
			ss << "Satellite directory " << opts.satelliteDir << " does not exist!";
			throw runtime_error(ss.str());
		}

		// Check if region exists
		if (!fs::exists(opts.regionDir / opts.region))
			throw runtime_error("No region found with name \"" + opts.region + "\"");
		fs::path clusterDir = opts.regionDir / opts.region / "BuildingClusters";
		if (!fs::exists(clusterDir) || fs::is_empty(clusterDir))
			throw runtime_error("No clusters for region \"" + opts.region + "\"");

		// Check if satellite images exist
		if (!fs::exists(opts.satelliteDir / opts.region) || fs::is_empty(opts.satelliteDir / opts.region))
			throw runtime_error("No satellite images for region \"" + opts.region + "\"");

		// Check if cluster masks exist
		fs::path clusterMaskDir = opts.dataDir / "regions" / opts.region / "clusterMasks" / opts.model;
		if (!fs::exists(clusterMaskDir) || fs::is_empty(clusterMaskDir))
			// TODO: generate cluster masks if needed
			throw runtime_error("No cluster masks for region \"" + opts.region + "\", model \"" + opts.model + "\"");

		// Create data directory if it doesn't exist
		if (!fs::exists(opts.dataDir))
			fs::create_directory(opts.dataDir);
		// Create region directory if it doesn't exist
		fs::path dataRegionDir = opts.dataDir / "regions" / opts.region;
		if (!fs::exists(dataRegionDir))
			fs::create_directories(dataRegionDir);

	// If not generating, make sure data directories already exist
	} else {
		if (!fs::exists(opts.dataDir)) {
			stringstream ss;
			ss << "Data directory " << opts.dataDir << " does not exist!";
			throw runtime_error(ss.str());
		}
		if (!fs::exists(opts.dataDir / "regions" / opts.region)) {
			stringstream ss;
			ss << "No data for region \"" << opts.region << "\"";
			throw runtime_error(ss.str());
		}
	}

	// Create output directories if they do not exist
	if (!fs::exists(opts.outputDir))
		fs::create_directory(opts.outputDir);
	if (!fs::exists(opts.outputDir / opts.region))
		fs::create_directory(opts.outputDir / opts.region);

	fs::path inputClusterDir = opts.regionDir / opts.region / "BuildingClusters";
	fs::path dataClusterDir = opts.dataDir / "regions" / opts.region;
	// Use all clusters in region if none specified
	if (opts.clusters.empty()) {
		fs::path clusterDir = opts.generate ? inputClusterDir : dataClusterDir;

		// Look for any integer-named directory to use as a cluster
		for (fs::directory_iterator di(clusterDir), dend; di != dend; ++di) {
			if (!fs::is_directory(di->path())) continue;
			// Only allow integer-named directories
			try {
				opts.clusters.insert(stoi(di->path().filename().string()));
			} catch (const exception& e) { continue; }
		}

	// Remove any non-existing clusters from cluster list
	} else {
		fs::path clusterDir = opts.generate ? inputClusterDir : dataClusterDir;

		for (auto it = opts.clusters.begin(); it != opts.clusters.end(); ) {
			// Get cluster ID string
			string cidStr; {
				stringstream ss; ss << setw(4) << setfill('0') << *it;
				cidStr = ss.str();
			}
			if (!fs::exists(clusterDir / cidStr) || fs::is_empty(clusterDir / cidStr)) {
				cout << "Cluster " << cidStr << " not found! Skipping..." << endl;
				it = opts.clusters.erase(it);
			} else
				++it;
		}
	}

	// Make sure we have some clusters left
	if (opts.clusters.empty()) {
		stringstream ss;
		ss << "No clusters to process...";
		if (!opts.generate)
			ss << endl << "Use -g to generate cluster data";
		throw runtime_error(ss.str());
	}
}

vector<Satellite> loadSatellites(Options& opts) {
	fs::path satDir = opts.satelliteDir / opts.region;
	vector<Satellite> sats;
	for (fs::directory_iterator di(satDir), dend; di != dend; ++di) {
		// Skip non-files and those not ending in .tif
		if (!fs::is_regular_file(di->path()) || di->path().extension() != ".tif")
			continue;
		// Create a Satellite object from file and add it to list
		cout << "Loading satellite " << di->path().stem().string() << "..." << endl;
		sats.push_back(Satellite(di->path()));
	}

	return sats;
}
