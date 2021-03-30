/*
 * This file is part of the VanitySearch distribution (https://github.com/JeanLucPons/VanitySearch).
 * Copyright (c) 2019 Jean Luc PONS.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef GPUENGINEH
#define GPUENGINEH

#include <vector>
#include "../SECP256k1.h"

#define SEARCH_COMPRESSED 0
#define SEARCH_UNCOMPRESSED 1
#define SEARCH_BOTH 2

// address mode
#define SINGLEMODE  0
#define FILEMODE   1

static const char* searchModes[] = { "Compressed", "Uncompressed", "Compressed or Uncompressed" };

// Number of key per thread (must be a multiple of GRP_SIZE) per kernel call
#define STEP_SIZE 1024

// Number of thread per block
#define ITEM_SIZE 28
#define ITEM_SIZE32 (ITEM_SIZE/4)
//#define _64K 65536

typedef struct {
	uint32_t thId;
	int16_t  incr;
	int16_t  endo;
	uint8_t* hash;
	bool mode;
} ITEM;

class GPUEngine
{

public:

	GPUEngine(int nbThreadGroup, int nbThreadPerGroup, int gpuId, uint32_t maxFound,
		int64_t BLOOM_SIZE, uint64_t BLOOM_BITS, uint8_t BLOOM_HASHES, const uint8_t* BLOOM_DATA,
		uint8_t* DATA, uint64_t TOTAL_ADDR);

	GPUEngine(int nbThreadGroup, int nbThreadPerGroup, int gpuId, uint32_t maxFound,
		uint32_t* hash160);

	~GPUEngine();
	bool SetKeys(Point* p);
	void SetSearchMode(int searchMode);
	void SetSearchType(int searchType);
	void SetAddressMode(int addressMode);

	bool Launch(std::vector<ITEM>& dataFound, bool spinWait = false);
	bool Launch2(std::vector<ITEM>& dataFound, bool spinWait = false);

	bool ClearOutBuffer();

	int GetNbThread();
	int GetGroupSize();

	//bool Check(Secp256K1 *secp);
	std::string deviceName;

	static void PrintCudaInfo();
	static void GenerateCode(Secp256K1* secp, int size);

private:

	bool callKernel();
	bool callKernel2();

	int CheckBinary(const uint8_t* hash);

	int nbThread;
	int nbThreadPerGroup;

	uint32_t* inputHash160;
	uint32_t* inputHash160Pinned;

	//uint8_t *bloomLookUp;
	uint8_t* inputBloomLookUp;
	uint8_t* inputBloomLookUpPinned;

	uint64_t* inputKey;
	uint64_t* inputKeyPinned;

	uint32_t* outputBuffer;
	uint32_t* outputBufferPinned;

	bool initialised;
	uint32_t searchMode;
	uint32_t searchType;
	uint32_t addressMode;
	bool littleEndian;

	//bool rekey;
	uint32_t maxFound;
	uint32_t outputSize;

	int64_t BLOOM_SIZE;
	uint64_t BLOOM_BITS;
	uint8_t BLOOM_HASHES;

	uint8_t* DATA;
	uint64_t TOTAL_ADDR;

};

#endif // GPUENGINEH
