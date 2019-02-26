#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <set>
#include <experimental/filesystem>
#include <rapidjson/document.h>
#include <rapidjson/prettywriter.h>
#include <rapidjson/istreamwrapper.h>
#include <rapidjson/ostreamwrapper.h>
#include "satellite.hpp"
#include "building.hpp"
#include "dn_predict.hpp"
using namespace std;
namespace rj = rapidjson;
namespace fs = std::experimental::filesystem;

// Program options
struct Options {
	// Commandline options
	string region;			// Which region to process (must be specified)
	set<string> clusters;	// Which clusters to process (empty -> all in region)
	string model;			// Which model to use (e.g., cgv_r, cgv_a, hist, etc.)
	bool all;				// Use all clusters in region
	bool noop;				// Do nothing (used for --help)
	fs::path configPath;	// Path to config file
	bool generate;			// Generate data from input directories

	// Config file options
	fs::path clusterDir;	// Dir containing clusters in a region
	fs::path satelliteDir;	// Dir containing pansharpened satellite imagery per region
	fs::path dataDir;		// Saved data output directory
	fs::path outputDir;		// Synthesized facades output directory

	// Constructor - set default values
	Options() :
		model("cgv_r"),
		all(false),
		noop(false),
		configPath("config.json"),
		generate(false) {}
};

// Functions
Options parseCmd(int argc, char** argv);
void readConfig(Options& opts);
void checkDirectories(Options& opts);
map<string, Satellite> loadSatellites(Options& opts);

// Program entry point
int main(int argc, char** argv) {
	try {
		// Parse commandline arguments
		Options opts = parseCmd(argc, argv);
		if (opts.noop) return 0;

		// Read config file
		readConfig(opts);

		// Check for input and output directories
		checkDirectories(opts);

		// Load satellite images if generating cluster data
		map<string, Satellite> sats;
		if (opts.generate)
			sats = loadSatellites(opts);

		// Load or generate each cluster
		vector<Building> bldgs;
		for (auto cluster : opts.clusters) {
			try {
				Building b;
				if (opts.generate) {
					cout << "Generating cluster " << cluster << "..." << endl;
					b.generate(opts.clusterDir, opts.dataDir, sats, opts.region, cluster, opts.model);
				} else {
					cout << "Loading cluster " << cluster << "..." << endl;
					b.load(opts.dataDir, opts.region, cluster, opts.model);
				}
				bldgs.push_back(std::move(b));
			} catch (const exception& e) {
				cout << "Failed to " << (opts.generate ? "generate" : "load") << " cluster "
					<< cluster << ": " << e.what() << endl;
				continue;
			}
		}

		// Generate cluster masks if they're missing
		fs::path clusterMaskDir = opts.dataDir / "clusterMasks" / opts.region / opts.model;
		if (!fs::exists(clusterMaskDir) && opts.all && opts.generate) {
			cout << "Generating cluster masks..." << endl;
			Building::createClusterMasks(opts.dataDir, sats, opts.region, opts.model, bldgs);
		}

		// Process each building
		for (auto& b : bldgs) {
			cout << "Processing cluster " << b.getCluster() << "..." << endl;
			try {
				// Score facades
				cout << "    Scoring facades..." << endl;
				map<size_t, fs::path> facadeMeta = b.scoreFacades(opts.outputDir);

				// Predict facade structure using DN
				cout << "    Predicting parameters..." << endl;
				for (auto& fi : facadeMeta)
					dn_predict(fi.second.string(), "model_config.json");

				// Synthesize facades
				cout << "    Synthesizing facades..." << endl;
				b.synthFacades(opts.outputDir, facadeMeta);
			} catch (const exception& e) {
				cout << "Failed to process cluster " << b.getCluster() << ": " << e.what() << endl;
				continue;
			}
		}

		// Combine all cluster outputs
		if (opts.all) {
			cout << "Combining all clusters..." << endl;
			Building::combineOutput(opts.outputDir, opts.region, opts.model, bldgs);
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
				int cid = stoi(arg);
				stringstream ss; ss << setw(4) << setfill('0') << cid;
				opts.clusters.insert(ss.str());
			// If failed, stop trying to get cluster IDs
			} catch (const exception& e) {
				break;
			}
		}
		// If no clusters specified, use all clusters
		if (opts.clusters.empty())
			opts.all = true;

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

// Loads the configuration file and stores its parameters into global variables
void readConfig(Options& opts) {
	try {
		// Read the config file
		ifstream configFile(opts.configPath);
		rj::IStreamWrapper isw(configFile);
		rj::Document config;
		config.ParseStream(isw);

		// Make sure cmdline options are configured
		fs::path inputDir = config["inputDir"].GetString();
		if (!config["regions"].HasMember(opts.region.c_str()))
			throw runtime_error("No region \"" + opts.region + "\" in config file!");
		if (!config["regions"][opts.region.c_str()]["models"].HasMember(opts.model.c_str()))
			throw runtime_error("No model \"" + opts.model + "\" in region \"" + opts.region
				+ "\" in config file!");

		// Read directories from config file
		opts.clusterDir = inputDir / config["regions"][opts.region.c_str()]
			["models"][opts.model.c_str()]["clusterDir"].GetString();
		opts.satelliteDir = inputDir / config["regions"][opts.region.c_str()]
			["satelliteDir"].GetString();
		opts.dataDir = config["dataDir"].GetString();
		opts.outputDir = config["outputDir"].GetString();

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
		if (!fs::exists(opts.clusterDir)) {
			stringstream ss;
			ss << "Cluster directory " << opts.clusterDir << " does not exist!";
			throw runtime_error(ss.str());
		}
		if (fs::is_empty(opts.clusterDir)) {
			stringstream ss;
			ss << "Cluster directory " << opts.clusterDir << " is empty!";
			throw runtime_error(ss.str());
		}
		if (!fs::exists(opts.satelliteDir)) {
			stringstream ss;
			ss << "Satellite directory " << opts.satelliteDir << " does not exist!";
			throw runtime_error(ss.str());
		}
		if (fs::is_empty(opts.satelliteDir)) {
			stringstream ss;
			ss << "Satellite directory " << opts.satelliteDir << " is empty!";
			throw runtime_error(ss.str());
		}

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
			ss << endl << "Use -g to generate cluster data";
			throw runtime_error(ss.str());
		}
		if (!fs::exists(opts.dataDir / "regions" / opts.region)) {
			stringstream ss;
			ss << "No data for region \"" << opts.region << "\"";
			ss << endl << "Use -g to generate cluster data";
			throw runtime_error(ss.str());
		}
	}

	// Create output directories if they do not exist
	if (!fs::exists(opts.outputDir))
		fs::create_directory(opts.outputDir);
	if (!fs::exists(opts.outputDir / opts.region))
		fs::create_directory(opts.outputDir / opts.region);

	fs::path dataClusterDir = opts.dataDir / "regions" / opts.region / opts.model;
	// Use all clusters in region if none specified
	if (opts.clusters.empty()) {
		fs::path clusterDir = opts.generate ? opts.clusterDir : dataClusterDir;

		// Look for any integer-named directory to use as a cluster
		for (fs::directory_iterator di(clusterDir), dend; di != dend; ++di) {
			if (!fs::is_directory(di->path())) continue;
			// Only allow integer-named directories
			try {
				stoi(di->path().filename().string());
				opts.clusters.insert(di->path().filename().string());
			} catch (const exception& e) { continue; }
		}

	// Remove any non-existing clusters from cluster list
	} else {
		fs::path clusterDir = opts.generate ? opts.clusterDir : dataClusterDir;

		for (auto it = opts.clusters.begin(); it != opts.clusters.end(); ) {
			// Get cluster ID string
			string cluster = *it;
			if (!fs::exists(clusterDir / cluster) || fs::is_empty(clusterDir / cluster)) {
				cout << "Cluster " << cluster << " not found! Skipping..." << endl;
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

// Reads all satellite datasets and returns a map of satellite name to Satellite object
map<string, Satellite> loadSatellites(Options& opts) {
	map<string, Satellite> sats;
	for (fs::directory_iterator di(opts.satelliteDir), dend; di != dend; ++di) {
		// Skip non-files and those not ending in .tif
		if (!fs::is_regular_file(di->path()) || di->path().extension() != ".tif")
			continue;

		// Create a Satellite object from file and add it to the map
		cout << "Loading satellite " << di->path().stem().string() << "..." << endl;
		Satellite sat(di->path());
		sats[sat.name] = std::move(sat);
	}

	return sats;
}
