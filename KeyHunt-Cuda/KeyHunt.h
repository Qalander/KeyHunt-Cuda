#ifndef KEYHUNTH
#define KEYHUNTH

#include <string>
#include <vector>
#include "SECP256k1.h"
#include "Bloom.h"
#include "GPU/GPUEngine.h"
#ifdef WIN64
#include <Windows.h>
#endif

#define CPU_GRP_SIZE (1024*2)

class KeyHunt;

typedef struct {
	KeyHunt* obj;
	int  threadId;
	bool isRunning;
	bool hasStarted;

	int  gridSizeX;
	int  gridSizeY;
	int  gpuId;

	Int rangeStart;
	Int rangeEnd;
	bool rKeyRequest;
} TH_PARAM;


class KeyHunt
{

public:

	KeyHunt(const std::string& inputFile, int compMode, int searchMode, int coinType, bool useGpu, 
		const std::string& outputFile, bool useSSE, uint32_t maxFound, uint64_t rKey, 
		const std::string& rangeStart, const std::string& rangeEnd, bool& should_exit);

	KeyHunt(const std::vector<unsigned char>& hashORxpoint, int compMode, int searchMode, int coinType, 
		bool useGpu, const std::string& outputFile, bool useSSE, uint32_t maxFound, uint64_t rKey, 
		const std::string& rangeStart, const std::string& rangeEnd, bool& should_exit);

	~KeyHunt();

	void Search(int nbThread, std::vector<int> gpuId, std::vector<int> gridSize, bool& should_exit);
	void FindKeyCPU(TH_PARAM* p);
	void FindKeyGPU(TH_PARAM* p);

private:

	void InitGenratorTable();

	std::string GetHex(std::vector<unsigned char>& buffer);
	bool checkPrivKey(std::string addr, Int& key, int32_t incr, bool mode);
	bool checkPrivKeyETH(std::string addr, Int& key, int32_t incr);
	bool checkPrivKeyX(Int& key, int32_t incr, bool mode);

	void checkMultiAddresses(bool compressed, Int key, int i, Point p1);
	void checkMultiAddressesETH(Int key, int i, Point p1);
	void checkSingleAddress(bool compressed, Int key, int i, Point p1);
	void checkSingleAddressETH(Int key, int i, Point p1);
	void checkMultiXPoints(bool compressed, Int key, int i, Point p1);
	void checkSingleXPoint(bool compressed, Int key, int i, Point p1);

	void checkMultiAddressesSSE(bool compressed, Int key, int i, Point p1, Point p2, Point p3, Point p4);
	void checkSingleAddressesSSE(bool compressed, Int key, int i, Point p1, Point p2, Point p3, Point p4);

	void output(std::string addr, std::string pAddr, std::string pAddrHex, std::string pubKey);
	bool isAlive(TH_PARAM* p);

	bool hasStarted(TH_PARAM* p);
	uint64_t getGPUCount();
	uint64_t getCPUCount();
	void rKeyRequest(TH_PARAM* p);
	void SetupRanges(uint32_t totalThreads);

	void getCPUStartingKey(Int& tRangeStart, Int& tRangeEnd, Int& key, Point& startP);
	void getGPUStartingKeys(Int& tRangeStart, Int& tRangeEnd, int groupSize, int nbThread, Int* keys, Point* p);

	int CheckBloomBinary(const uint8_t* _xx, uint32_t K_LENGTH);
	bool MatchHash(uint32_t* _h);
	bool MatchXPoint(uint32_t* _h);
	std::string formatThousands(uint64_t x);
	char* toTimeStr(int sec, char* timeStr);

	Secp256K1* secp;
	Bloom* bloom;

	uint64_t counters[256];
	double startTime;

	int compMode;
	int searchMode;
	int coinType;

	bool useGpu;
	bool endOfSearch;
	int nbCPUThread;
	int nbGPUThread;
	int nbFoundKey;
	uint64_t targetCounter;

	std::string outputFile;
	std::string inputFile;
	uint32_t hash160Keccak[5];
	uint32_t xpoint[8];
	bool useSSE;

	Int rangeStart;
	Int rangeEnd;
	Int rangeDiff;
	Int rangeDiff2;

	uint32_t maxFound;
	uint64_t rKey;
	uint64_t lastrKey;

	uint8_t* DATA;
	uint64_t TOTAL_COUNT;
	uint64_t BLOOM_N;

#ifdef WIN64
	HANDLE ghMutex;
#else
	pthread_mutex_t  ghMutex;
#endif

};

#endif // KEYHUNTH
