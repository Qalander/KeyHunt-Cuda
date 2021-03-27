#include "Timer.h"
#include "KeyHunt.h"
#include "ArgParse.h"
#include <fstream>
#include <string>
#include <string.h>
#include <stdexcept>

#define RELEASE "1.03"

using namespace std;
using namespace argparse;
bool should_exit = false;

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------

const char* vstr = "Print version                                                                                   ";
const char* cstr = "Check the working of the codes                                                                  ";
const char* ustr = "Search uncompressed addresses                                                                   ";
const char* bstr = "Search both uncompressed or compressed addresses                                                ";
const char* gstr = "Enable GPU calculation                                                                          ";
const char* istr = "GPU ids: 0,1...: List of GPU(s) to use, default is 0                                            ";
const char* xstr = "GPU gridsize: g0x,g0y,g1x,g1y, ...: Specify GPU(s) kernel gridsize, default is 8*(MP number),128";
const char* ostr = "Outputfile: Output results to the specified file, default: Found.txt                            ";
const char* mstr = "Specify maximun number of addresses found by each kernel call                                   ";
const char* sstr = "Seed: Specify a seed for the base key, default is random                                        ";
const char* tstr = "threadNumber: Specify number of CPU thread, default is number of core                           ";
const char* estr = "Disable SSE hash function                                                                       ";
const char* lstr = "List cuda enabled devices                                                                       ";
const char* rstr = "Rkey: Rekey interval in MegaKey, default is disabled                                            ";
const char* nstr = "Number of base key random bits                                                                  ";
const char* fstr = "RIPEMD160 binary hash file path                                                                 ";


// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------

void getInts(string name, vector<int>& tokens, const string& text, char sep)
{

	size_t start = 0, end = 0;
	tokens.clear();
	int item;

	try {

		while ((end = text.find(sep, start)) != string::npos) {
			item = std::stoi(text.substr(start, end - start));
			tokens.push_back(item);
			start = end + 1;
		}

		item = std::stoi(text.substr(start));
		tokens.push_back(item);

	}
	catch (std::invalid_argument&) {

		printf("Invalid %s argument, number expected\n", name.c_str());
		exit(-1);

	}

}

BOOL WINAPI CtrlHandler(DWORD fdwCtrlType)
{
	switch (fdwCtrlType) {
	case CTRL_C_EVENT:
		//printf("\n\nCtrl-C event\n\n");
		should_exit = true;
		return TRUE;

	default:
		return TRUE;
	}
}

int main(int argc, const char* argv[])
{
	// Global Init
	Timer::Init();
	rseed(Timer::getSeed32());

	bool gpuEnable = false;
	int searchMode = SEARCH_COMPRESSED;
	vector<int> gpuId = { 0 };
	vector<int> gridSize;
	string seed = "";
	string outputFile = "Found.txt";
	string hash160File = "";
	int nbCPUThread = Timer::getCoreNumber();
	int nbit = 0;
	bool tSpecified = false;
	bool sse = true;
	uint32_t maxFound = 1024 * 64;
	uint64_t rekey = 0;
	bool paranoiacSeed = false;

	ArgumentParser parser("KeyHunt-Cuda", "Hunt for Bitcoin private keys.");

	parser.add_argument("-v", "--version", vstr, false);
	parser.add_argument("-c", "--check", cstr, false);
	parser.add_argument("-u", "--uncomp", ustr, false);
	parser.add_argument("-b", "--both", bstr, false);
	parser.add_argument("-g", "--gpu", gstr, false);
	parser.add_argument("-i", "--gpui", istr, false);
	parser.add_argument("-x", "--gpux", xstr, false);
	parser.add_argument("-o", "--out", ostr, false);
	parser.add_argument("-m", "--max", mstr, false);
	parser.add_argument("-s", "--seed", sstr, false);
	parser.add_argument("-t", "--thread", tstr, false);
	parser.add_argument("-e", "--nosse", estr, false);
	parser.add_argument("-l", "--list", lstr, false);
	parser.add_argument("-r", "--rkey", rstr, false);
	parser.add_argument("-n", "--nbit", nstr, false);
	parser.add_argument("-f", "--file", fstr, false);
	parser.enable_help();

	auto err = parser.parse(argc, argv);
	if (err) {
		std::cout << err << std::endl;
		parser.print_help();
		return -1;
	}

	if (parser.exists("help")) {
		parser.print_help();
		return 0;
	}

	if (parser.exists("version")) {
		printf("KeyHunt-Cuda v" RELEASE "\n");
		return 0;
	}

	if (parser.exists("check")) {
		printf("KeyHunt-Cuda v" RELEASE "\n");
		printf("Checking... Int\n\n");
		Int K;
		K.SetBase16("3EF7CEF65557B61DC4FF2313D0049C584017659A32B002C105D04A19DA52CB47");
		K.Check();

		printf("\n\nChecking... Secp256K1\n\n");
		Secp256K1 sec;
		sec.Init();
		sec.Check();

		return 0;
	}

	if (parser.exists("uncomp")) {
		searchMode = SEARCH_UNCOMPRESSED;
	}
	if (parser.exists("both")) {
		searchMode = SEARCH_BOTH;
	}

	if (parser.exists("gpu")) {
		gpuEnable = true;
	}

	if (parser.exists("gpui")) {
		string ids = parser.get<string>("i");
		getInts("gpui", gpuId, ids, ',');
	}

	if (parser.exists("gpux")) {
		string grids = parser.get<string>("x");
		getInts("gpux", gridSize, grids, ',');
	}

	if (parser.exists("out")) {
		outputFile = parser.get<string>("o");
	}

	if (parser.exists("max")) {
		maxFound = parser.get<uint32_t>("m");
	}

	if (parser.exists("seed")) {
		seed = parser.get<string>("s");
		paranoiacSeed = true;
	}

	if (parser.exists("thread")) {
		nbCPUThread = parser.get<int>("t");
		tSpecified = true;
	}

	if (parser.exists("nosse")) {
		sse = false;
	}

	if (parser.exists("list")) {
		GPUEngine::PrintCudaInfo();
		return 0;
	}

	if (parser.exists("rkey")) {
		rekey = parser.get<uint64_t>("r");
	}

	if (parser.exists("nbit")) {
		nbit = parser.get<int>("n");
		if (nbit < 0 || nbit > 256) {
			printf("Invalid nbit value, must have in range: 1 - 256\n");
			exit(-1);
		}
	}

	if (parser.exists("file")) {
		hash160File = parser.get<string>("f");
	}


	if (gridSize.size() == 0) {
		for (int i = 0; i < gpuId.size(); i++) {
			gridSize.push_back(-1);
			gridSize.push_back(128);
		}
	}
	else if (gridSize.size() != gpuId.size() * 2) {
		printf("Invalid gridSize or gpuId argument, must have coherent size\n");
		exit(-1);
	}

	if (hash160File.length() <= 0) {
		printf("Invalid RIPEMD160 binary hash file path\n");
		exit(-1);
	}

	// Let one CPU core free per gpu is gpu is enabled
	// It will avoid to hang the system
	if (!tSpecified && nbCPUThread > 1 && gpuEnable)
		nbCPUThread -= (int)gpuId.size();
	if (nbCPUThread < 0)
		nbCPUThread = 0;

	{
		printf("\n");
		printf("KeyHunt-Cuda v" RELEASE "\n");
		printf("\n");
		printf("MODE         : %s\n", searchMode == SEARCH_COMPRESSED ? "COMPRESSED" : (searchMode == SEARCH_UNCOMPRESSED ? "UNCOMPRESSED" : "COMPRESSED & UNCOMPRESSED"));
		printf("DEVICE       : %s\n", (gpuEnable && nbCPUThread > 0) ? "CPU & GPU" : ((!gpuEnable && nbCPUThread > 0) ? "CPU" : "GPU"));
		printf("CPU THREAD   : %d\n", nbCPUThread);
		printf("GPU IDS      : ");
		for (int i = 0; i < gpuId.size(); i++) {
			printf("%d", gpuId.at(i));
			if (i + 1 < gpuId.size())
				printf(", ");
		}
		printf("\n");
		printf("GPU GRIDSIZE : ");
		for (int i = 0; i < gridSize.size(); i++) {
			printf("%d", gridSize.at(i));
			if (i + 1 < gridSize.size()) {
				if ((i + 1) % 2 != 0) {
					printf("x");
				}
				else {
					printf(", ");
				}

			}
		}
		printf("\n");
		printf("SSE          : %s\n", sse ? "YES" : "NO");
		printf("SEED         : %s\n", seed.c_str());
		printf("RKEY(Mk)     : %llu\n", rekey);
		printf("NBIT         : %d\n", nbit);
		printf("MAX FOUND    : %d\n", maxFound);
		printf("HASH160 FILE : %s\n", hash160File.c_str());
		printf("OUTPUT FILE  : %s\n", outputFile.c_str());
	}

	if (SetConsoleCtrlHandler(CtrlHandler, TRUE)) {
		KeyHunt* v = new KeyHunt(hash160File, seed, searchMode, gpuEnable,
			outputFile, sse, maxFound, rekey, nbit, paranoiacSeed, should_exit);

		v->Search(nbCPUThread, gpuId, gridSize, should_exit);

		delete v;
		printf("\n\nBYE\n");
		return 0;
	}
	else {
		printf("error: could not set control-c handler\n");
		return 1;
	}
}