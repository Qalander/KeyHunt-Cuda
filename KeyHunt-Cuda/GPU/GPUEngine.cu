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

#include "GPUEngine.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdint.h>
#include "../hash/sha256.h"
#include "../hash/ripemd160.h"
#include "../Timer.h"

#include "GPUGroup.h"
#include "GPUMath.h"
#include "GPUHash.h"
#include "GPUBase58.h"
#include "GPUCompute.h"

// ---------------------------------------------------------------------------------------
#define CudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )

inline void __cudaSafeCall(cudaError err, const char* file, const int line)
{
	if (cudaSuccess != err)
	{
		fprintf(stderr, "cudaSafeCall() failed at %s:%i : %s\n", file, line, cudaGetErrorString(err));
		exit(-1);
	}
	return;
}

// ---------------------------------------------------------------------------------------

__global__ void comp_keys(uint32_t mode, uint8_t* bloomLookUp, int BLOOM_BITS, uint8_t BLOOM_HASHES,
	uint64_t* keys, uint32_t maxFound, uint32_t* found)
{

	int xPtr = (blockIdx.x * blockDim.x) * 8;
	int yPtr = xPtr + 4 * blockDim.x;
	ComputeKeys(mode, keys + xPtr, keys + yPtr, bloomLookUp, BLOOM_BITS, BLOOM_HASHES, maxFound, found);

}

__global__ void comp_keys_comp(uint8_t* bloomLookUp, int BLOOM_BITS, uint8_t BLOOM_HASHES, uint64_t* keys,
	uint32_t maxFound, uint32_t* found)
{

	int xPtr = (blockIdx.x * blockDim.x) * 8;
	int yPtr = xPtr + 4 * blockDim.x;
	ComputeKeysComp(keys + xPtr, keys + yPtr, bloomLookUp, BLOOM_BITS, BLOOM_HASHES, maxFound, found);

}

__global__ void clear_counter(uint32_t* found)
{
	ClearCouter(found);
}

// ---------------------------------------------------------------------------------------

using namespace std;

int _ConvertSMVer2Cores(int major, int minor)
{

	// Defines for GPU Architecture types (using the SM version to determine
	// the # of cores per SM
	typedef struct {
		int SM;  // 0xMm (hexidecimal notation), M = SM Major version,
		// and m = SM minor version
		int Cores;
	} sSMtoCores;

	sSMtoCores nGpuArchCoresPerSM[] = {
		{0x20, 32}, // Fermi Generation (SM 2.0) GF100 class
		{0x21, 48}, // Fermi Generation (SM 2.1) GF10x class
		{0x30, 192},
		{0x32, 192},
		{0x35, 192},
		{0x37, 192},
		{0x50, 128},
		{0x52, 128},
		{0x53, 128},
		{0x60,  64},
		{0x61, 128},
		{0x62, 128},
		{0x70,  64},
		{0x72,  64},
		{0x75,  64},
		{-1, -1}
	};

	int index = 0;

	while (nGpuArchCoresPerSM[index].SM != -1) {
		if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor)) {
			return nGpuArchCoresPerSM[index].Cores;
		}

		index++;
	}

	return 0;

}

GPUEngine::GPUEngine(int nbThreadGroup, int nbThreadPerGroup, int gpuId, uint32_t maxFound, bool rekey,
	int64_t BLOOM_SIZE, uint64_t BLOOM_BITS, uint8_t BLOOM_HASHES, const uint8_t* BLOOM_DATA,
	uint8_t* DATA, uint64_t TOTAL_ADDR)
{

	// Initialise CUDA
	this->rekey = rekey;
	this->nbThreadPerGroup = nbThreadPerGroup;

	this->BLOOM_SIZE = BLOOM_SIZE;
	this->BLOOM_BITS = BLOOM_BITS;
	this->BLOOM_HASHES = BLOOM_HASHES;
	this->DATA = DATA;
	this->TOTAL_ADDR = TOTAL_ADDR;

	initialised = false;

	int deviceCount = 0;
	CudaSafeCall(cudaGetDeviceCount(&deviceCount));

	// This function call returns 0 if there are no CUDA capable devices.
	if (deviceCount == 0) {
		printf("GPUEngine: There are no available device(s) that support CUDA\n");
		return;
	}

	CudaSafeCall(cudaSetDevice(gpuId));

	cudaDeviceProp deviceProp;
	CudaSafeCall(cudaGetDeviceProperties(&deviceProp, gpuId));

	if (nbThreadGroup == -1)
		nbThreadGroup = deviceProp.multiProcessorCount * 8;

	this->nbThread = nbThreadGroup * nbThreadPerGroup;
	this->maxFound = maxFound;
	this->outputSize = (maxFound * ITEM_SIZE + 4);

	char tmp[512];
	sprintf(tmp, "GPU #%d %s (%dx%d cores) Grid(%dx%d)",
		gpuId, deviceProp.name, deviceProp.multiProcessorCount,
		_ConvertSMVer2Cores(deviceProp.major, deviceProp.minor),
		nbThread / nbThreadPerGroup,
		nbThreadPerGroup);
	deviceName = std::string(tmp);

	// Prefer L1 (We do not use __shared__ at all)
	CudaSafeCall(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));

	size_t stackSize = 49152;
	CudaSafeCall(cudaDeviceSetLimit(cudaLimitStackSize, stackSize));

	// Allocate memory
	CudaSafeCall(cudaMalloc((void**)&inputKey, nbThread * 32 * 2));
	CudaSafeCall(cudaHostAlloc(&inputKeyPinned, nbThread * 32 * 2, cudaHostAllocWriteCombined | cudaHostAllocMapped));

	CudaSafeCall(cudaMalloc((void**)&outputBuffer, outputSize));
	CudaSafeCall(cudaHostAlloc(&outputBufferPinned, outputSize, cudaHostAllocWriteCombined | cudaHostAllocMapped));

	CudaSafeCall(cudaMalloc((void**)&inputBloomLookUp, BLOOM_SIZE));
	CudaSafeCall(cudaHostAlloc(&inputBloomLookUpPinned, BLOOM_SIZE, cudaHostAllocWriteCombined | cudaHostAllocMapped));

	memcpy(inputBloomLookUpPinned, BLOOM_DATA, BLOOM_SIZE);

	CudaSafeCall(cudaMemcpy(inputBloomLookUp, inputBloomLookUpPinned, BLOOM_SIZE, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaFreeHost(inputBloomLookUpPinned));
	inputBloomLookUpPinned = NULL;

	CudaSafeCall(cudaGetLastError());

	searchMode = SEARCH_COMPRESSED;
	searchType = P2PKH;
	initialised = true;

}

int GPUEngine::GetGroupSize()
{
	return GRP_SIZE;
}

void GPUEngine::PrintCudaInfo()
{
	const char* sComputeMode[] = {
		"Multiple host threads",
		"Only one host thread",
		"No host thread",
		"Multiple process threads",
		"Unknown",
		NULL
	};

	int deviceCount = 0;
	CudaSafeCall(cudaGetDeviceCount(&deviceCount));

	// This function call returns 0 if there are no CUDA capable devices.
	if (deviceCount == 0) {
		printf("GPUEngine: There are no available device(s) that support CUDA\n");
		return;
	}

	for (int i = 0; i < deviceCount; i++) {
		CudaSafeCall(cudaSetDevice(i));
		cudaDeviceProp deviceProp;
		CudaSafeCall(cudaGetDeviceProperties(&deviceProp, i));
		printf("GPU #%d %s (%dx%d cores) (Cap %d.%d) (%.1f MB) (%s)\n",
			i, deviceProp.name, deviceProp.multiProcessorCount,
			_ConvertSMVer2Cores(deviceProp.major, deviceProp.minor),
			deviceProp.major, deviceProp.minor, (double)deviceProp.totalGlobalMem / 1048576.0,
			sComputeMode[deviceProp.computeMode]);
	}
}

GPUEngine::~GPUEngine()
{
	CudaSafeCall(cudaFree(inputKey));
	CudaSafeCall(cudaFree(inputBloomLookUp));
	CudaSafeCall(cudaFreeHost(outputBufferPinned));
	CudaSafeCall(cudaFree(outputBuffer));
}

int GPUEngine::GetNbThread()
{
	return nbThread;
}

void GPUEngine::SetSearchMode(int searchMode)
{
	this->searchMode = searchMode;
}

void GPUEngine::SetSearchType(int searchType)
{
	this->searchType = searchType;
}

bool GPUEngine::callKernel()
{

	// Reset nbFound
	CudaSafeCall(cudaMemset(outputBuffer, 0, 4));

	// Call the kernel (Perform STEP_SIZE keys per thread)
	if (searchType == P2PKH) {
		if (searchMode == SEARCH_COMPRESSED) {
			comp_keys_comp << < nbThread / nbThreadPerGroup, nbThreadPerGroup >> >
				(inputBloomLookUp, BLOOM_BITS, BLOOM_HASHES, inputKey, maxFound, outputBuffer);
		}
		else {
			comp_keys << < nbThread / nbThreadPerGroup, nbThreadPerGroup >> >
				(searchMode, inputBloomLookUp, BLOOM_BITS, BLOOM_HASHES, inputKey, maxFound, outputBuffer);
		}
	}
	else {
		printf("GPUEngine: Wrong searchType\n");
		return false;
	}

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("GPUEngine: Kernel: %s\n", cudaGetErrorString(err));
		return false;
	}
	return true;

}

bool GPUEngine::ClearOutBuffer()
{
	clear_counter << < nbThread / nbThreadPerGroup, nbThreadPerGroup >> > (outputBuffer);

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("GPUEngine: ClearOutBuffer: %s\n", cudaGetErrorString(err));
		return false;
	}
	return true;
}

bool GPUEngine::SetKeys(Point* p)
{

	// Sets the starting keys for each thread
	// p must contains nbThread public keys
	for (int i = 0; i < nbThread; i += nbThreadPerGroup) {
		for (int j = 0; j < nbThreadPerGroup; j++) {

			inputKeyPinned[8 * i + j + 0 * nbThreadPerGroup] = p[i + j].x.bits64[0];
			inputKeyPinned[8 * i + j + 1 * nbThreadPerGroup] = p[i + j].x.bits64[1];
			inputKeyPinned[8 * i + j + 2 * nbThreadPerGroup] = p[i + j].x.bits64[2];
			inputKeyPinned[8 * i + j + 3 * nbThreadPerGroup] = p[i + j].x.bits64[3];

			inputKeyPinned[8 * i + j + 4 * nbThreadPerGroup] = p[i + j].y.bits64[0];
			inputKeyPinned[8 * i + j + 5 * nbThreadPerGroup] = p[i + j].y.bits64[1];
			inputKeyPinned[8 * i + j + 6 * nbThreadPerGroup] = p[i + j].y.bits64[2];
			inputKeyPinned[8 * i + j + 7 * nbThreadPerGroup] = p[i + j].y.bits64[3];

		}
	}

	// Fill device memory
	CudaSafeCall(cudaMemcpy(inputKey, inputKeyPinned, nbThread * 32 * 2, cudaMemcpyHostToDevice));

	if (!rekey) {
		// We do not need the input pinned memory anymore
		CudaSafeCall(cudaFreeHost(inputKeyPinned));
		inputKeyPinned = NULL;
	}

	return callKernel();
}

bool GPUEngine::Launch(std::vector<ITEM>& dataFound, bool spinWait)
{
	dataFound.clear();


	// Get the result
	if (spinWait) {
		CudaSafeCall(cudaMemcpy(outputBufferPinned, outputBuffer, outputSize, cudaMemcpyDeviceToHost));
	}
	else {
		// Use cudaMemcpyAsync to avoid default spin wait of cudaMemcpy wich takes 100% CPU
		cudaEvent_t evt;
		CudaSafeCall(cudaEventCreate(&evt));
		CudaSafeCall(cudaMemcpyAsync(outputBufferPinned, outputBuffer, 4, cudaMemcpyDeviceToHost, 0));
		CudaSafeCall(cudaEventRecord(evt, 0));
		while (cudaEventQuery(evt) == cudaErrorNotReady) {
			// Sleep 1 ms to free the CPU
			Timer::SleepMillis(1);
		}
		CudaSafeCall(cudaEventDestroy(evt));
	}

	// Look for prefix found
	uint32_t nbFound = outputBufferPinned[0];
	if (nbFound > maxFound) {
		nbFound = maxFound;
	}

	// When can perform a standard copy, the kernel is eneded
	CudaSafeCall(cudaMemcpy(outputBufferPinned, outputBuffer, nbFound * ITEM_SIZE + 4, cudaMemcpyDeviceToHost));

	for (uint32_t i = 0; i < nbFound; i++) {

		uint32_t* itemPtr = outputBufferPinned + (i * ITEM_SIZE32 + 1);
		uint8_t* hash = (uint8_t*)(itemPtr + 2);
		if (CheckBinary(hash) > 0) {

			ITEM it;
			it.thId = itemPtr[0];
			int16_t* ptr = (int16_t*)&(itemPtr[1]);
			it.endo = ptr[0] & 0x7FFF;
			it.mode = (ptr[0] & 0x8000) != 0;
			it.incr = ptr[1];
			it.hash = (uint8_t*)(itemPtr + 2);
			dataFound.push_back(it);
		}
	}
	return callKernel();
}

int GPUEngine::CheckBinary(const uint8_t* hash)
{
	uint8_t* temp_read;
	uint64_t half, min, max, current; //, current_offset
	int64_t rcmp;
	int32_t r = 0;
	min = 0;
	current = 0;
	max = TOTAL_ADDR;
	half = TOTAL_ADDR;
	while (!r && half >= 1) {
		half = (max - min) / 2;
		temp_read = DATA + ((current + half) * 20);
		rcmp = memcmp(hash, temp_read, 20);
		if (rcmp == 0) {
			r = 1;  //Found!!
		}
		else {
			if (rcmp < 0) { //data < temp_read
				max = (max - half);
			}
			else { // data > temp_read
				min = (min + half);
			}
			current = min;
		}
	}
	return r;
}




