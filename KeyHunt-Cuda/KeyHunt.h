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

#define CPU_GRP_SIZE 1024

class KeyHunt;

typedef struct {

	KeyHunt* obj;
	int  threadId;
	bool isRunning;
	bool hasStarted;
	bool rekeyRequest;
	int  gridSizeX;
	int  gridSizeY;
	int  gpuId;

} TH_PARAM;


class KeyHunt
{

public:

	KeyHunt(std::string addressFile, std::string seed, int searchMode,
		bool useGpu, std::string outputFile, bool useSSE, uint32_t maxFound,
		uint64_t rekey, bool paranoiacSeed, bool& should_exit);
	~KeyHunt();

	void Search(int nbThread, std::vector<int> gpuId, std::vector<int> gridSize, bool& should_exit);
	void FindKeyCPU(TH_PARAM* p);
	void FindKeyGPU(TH_PARAM* p);

private:

	std::string GetHex(std::vector<unsigned char>& buffer);
	bool checkPrivKey(std::string addr, Int& key, int32_t incr, int endomorphism, bool mode);
	void checkAddresses(bool compressed, Int key, int i, Point p1);
	void checkAddressesSSE(bool compressed, Int key, int i, Point p1, Point p2, Point p3, Point p4);
	void output(std::string addr, std::string pAddr, std::string pAddrHex);
	bool isAlive(TH_PARAM* p);

	bool hasStarted(TH_PARAM* p);
	void rekeyRequest(TH_PARAM* p);
	uint64_t getGPUCount();
	uint64_t getCPUCount();

	void getCPUStartingKey(int thId, Int& key, Point& startP);
	void getGPUStartingKeys(int thId, int groupSize, int nbThread, Int* keys, Point* p);

	int CheckBloomBinary(const uint8_t* hash);
	std::string formatThousands(uint64_t x);
	char* toTimeStr(int sec, char* timeStr);

	Secp256K1* secp;
	Bloom* bloom;

	Int startKey;
	uint64_t counters[256];
	double startTime;

	int searchMode;
	int searchType;

	bool useGpu;
	bool endOfSearch;
	int nbCPUThread;
	int nbGPUThread;
	int nbFoundKey;
	uint64_t rekey;
	uint64_t lastRekey;
	std::string outputFile;
	std::string addressFile;
	bool useSSE;

	uint32_t maxFound;

	uint8_t* DATA;
	uint64_t TOTAL_ADDR;
	uint64_t BLOOM_N;

	Int beta;
	Int lambda;
	Int beta2;
	Int lambda2;

#ifdef WIN64
	HANDLE ghMutex;
#else
	pthread_mutex_t  ghMutex;
#endif

};

#endif // KEYHUNTH
