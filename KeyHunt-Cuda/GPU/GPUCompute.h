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

#include <device_atomic_functions.h>
#include <device_functions.h>

__device__ uint64_t* _2Gnx = NULL;
__device__ uint64_t* _2Gny = NULL;

__device__ uint64_t* Gx = NULL;
__device__ uint64_t* Gy = NULL;

// ---------------------------------------------------------------------------------------

__device__ int Test_Bit_Set_Bit(const uint8_t* buf, uint32_t bit)
{
	uint32_t byte = bit >> 3;
	uint8_t c = buf[byte];        // expensive memory access
	uint8_t mask = 1 << (bit % 8);

	if (c & mask) {
		return 1;
	}
	else {
		return 0;
	}
}

// ---------------------------------------------------------------------------------------

__device__ uint32_t MurMurHash2(const void* key, int len, uint32_t seed)
{
	const uint32_t m = 0x5bd1e995;
	const int r = 24;

	uint32_t h = seed ^ len;
	const uint8_t* data = (const uint8_t*)key;
	while (len >= 4) {
		uint32_t k = *(uint32_t*)data;
		k *= m;
		k ^= k >> r;
		k *= m;
		h *= m;
		h ^= k;
		data += 4;
		len -= 4;
	}
	switch (len) {
	case 3: h ^= data[2] << 16;
		break;
	case 2: h ^= data[1] << 8;
		break;
	case 1: h ^= data[0];
		h *= m;
		break;
	}

	h ^= h >> 13;
	h *= m;
	h ^= h >> 15;

	return h;
}

// ---------------------------------------------------------------------------------------

__device__ int BloomCheck(const uint32_t* hash, const uint8_t* inputBloomLookUp, uint64_t BLOOM_BITS, uint8_t BLOOM_HASHES, uint32_t K_LENGTH)
{
	int add = 0;
	uint8_t hits = 0;
	uint32_t a = MurMurHash2((uint8_t*)hash, K_LENGTH, 0x9747b28c);
	uint32_t b = MurMurHash2((uint8_t*)hash, K_LENGTH, a);
	uint32_t x;
	uint8_t i;
	for (i = 0; i < BLOOM_HASHES; i++) {
		x = (a + b * i) % BLOOM_BITS;
		if (Test_Bit_Set_Bit(inputBloomLookUp, x)) {
			hits++;
		}
		else if (!add) {
			return 0;
		}
	}
	if (hits == BLOOM_HASHES) {
		return 1;
	}
	return 0;
}

// ---------------------------------------------------------------------------------------

__device__ __noinline__ void CheckPointSEARCH_MODE_MA(uint32_t* _h, int32_t incr, int32_t mode,
	uint8_t* bloomLookUp, uint64_t BLOOM_BITS, uint8_t BLOOM_HASHES, uint32_t maxFound, uint32_t* out)
{
	uint32_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (BloomCheck(_h, bloomLookUp, BLOOM_BITS, BLOOM_HASHES, 20) > 0) {
		uint32_t pos = atomicAdd(out, 1);
		if (pos < maxFound) {
			out[pos * ITEM_SIZE_A32 + 1] = tid;
			out[pos * ITEM_SIZE_A32 + 2] = (uint32_t)(incr << 16) | (uint32_t)(mode << 15);// | (uint32_t)(endo);
			out[pos * ITEM_SIZE_A32 + 3] = _h[0];
			out[pos * ITEM_SIZE_A32 + 4] = _h[1];
			out[pos * ITEM_SIZE_A32 + 5] = _h[2];
			out[pos * ITEM_SIZE_A32 + 6] = _h[3];
			out[pos * ITEM_SIZE_A32 + 7] = _h[4];
		}
	}
}

// ---------------------------------------------------------------------------------------

__device__ __noinline__ void CheckPointSEARCH_MODE_MX(uint32_t* _h, int32_t incr, int32_t mode,
	uint8_t* bloomLookUp, uint64_t BLOOM_BITS, uint8_t BLOOM_HASHES, uint32_t maxFound, uint32_t* out)
{
	uint32_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (BloomCheck(_h, bloomLookUp, BLOOM_BITS, BLOOM_HASHES, 32) > 0) {
		uint32_t pos = atomicAdd(out, 1);
		if (pos < maxFound) {
			out[pos * ITEM_SIZE_X32 + 1] = tid;
			out[pos * ITEM_SIZE_X32 + 2] = (uint32_t)(incr << 16) | (uint32_t)(mode << 15);// | (uint32_t)(endo);
			out[pos * ITEM_SIZE_X32 + 3] = _h[0];
			out[pos * ITEM_SIZE_X32 + 4] = _h[1];
			out[pos * ITEM_SIZE_X32 + 5] = _h[2];
			out[pos * ITEM_SIZE_X32 + 6] = _h[3];
			out[pos * ITEM_SIZE_X32 + 7] = _h[4];
			out[pos * ITEM_SIZE_X32 + 8] = _h[5];
			out[pos * ITEM_SIZE_X32 + 9] = _h[6];
			out[pos * ITEM_SIZE_X32 + 10] = _h[7];
		}
	}
}

// ---------------------------------------------------------------------------------------

__device__ __noinline__ bool MatchHash(uint32_t* _h, uint32_t* hash)
{
	if (_h[0] == hash[0] &&
		_h[1] == hash[1] &&
		_h[2] == hash[2] &&
		_h[3] == hash[3] &&
		_h[4] == hash[4]) {
		return true;
	}
	else {
		return false;
	}
}

// ---------------------------------------------------------------------------------------

__device__ __noinline__ bool MatchXPoint(uint32_t* _h, uint32_t* xpoint)
{
	//for (int i = 0; i < 32; i++) {
	//	printf("%02x", ((uint8_t*)xpoint)[i]);
	//}
	//printf("\n");

	if (_h[0] == xpoint[0] &&
		_h[1] == xpoint[1] &&
		_h[2] == xpoint[2] &&
		_h[3] == xpoint[3] &&
		_h[4] == xpoint[4] &&
		_h[5] == xpoint[5] &&
		_h[6] == xpoint[6] &&
		_h[7] == xpoint[7]) {
		return true;
	}
	else {
		return false;
	}
}

// ---------------------------------------------------------------------------------------

__device__ __noinline__ void CheckPointSEARCH_MODE_SA(uint32_t* _h, int32_t incr, int32_t mode,
	uint32_t* hash160, uint32_t maxFound, uint32_t* out)
{
	uint32_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (MatchHash(_h, hash160)) {
		uint32_t pos = atomicAdd(out, 1);
		if (pos < maxFound) {
			out[pos * ITEM_SIZE_A32 + 1] = tid;
			out[pos * ITEM_SIZE_A32 + 2] = (uint32_t)(incr << 16) | (uint32_t)(mode << 15);// | (uint32_t)(endo);
			out[pos * ITEM_SIZE_A32 + 3] = _h[0];
			out[pos * ITEM_SIZE_A32 + 4] = _h[1];
			out[pos * ITEM_SIZE_A32 + 5] = _h[2];
			out[pos * ITEM_SIZE_A32 + 6] = _h[3];
			out[pos * ITEM_SIZE_A32 + 7] = _h[4];
		}
	}
}

// ---------------------------------------------------------------------------------------

__device__ __noinline__ void CheckPointSEARCH_MODE_SX(uint32_t* _h, int32_t incr, int32_t mode,
	uint32_t* xpoint, uint32_t maxFound, uint32_t* out)
{
	uint32_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (MatchXPoint(_h, xpoint)) {
		uint32_t pos = atomicAdd(out, 1);
		if (pos < maxFound) {
			out[pos * ITEM_SIZE_X32 + 1] = tid;
			out[pos * ITEM_SIZE_X32 + 2] = (uint32_t)(incr << 16) | (uint32_t)(mode << 15);// | (uint32_t)(endo);
			out[pos * ITEM_SIZE_X32 + 3] = _h[0];
			out[pos * ITEM_SIZE_X32 + 4] = _h[1];
			out[pos * ITEM_SIZE_X32 + 5] = _h[2];
			out[pos * ITEM_SIZE_X32 + 6] = _h[3];
			out[pos * ITEM_SIZE_X32 + 7] = _h[4];
			out[pos * ITEM_SIZE_X32 + 8] = _h[5];
			out[pos * ITEM_SIZE_X32 + 9] = _h[6];
			out[pos * ITEM_SIZE_X32 + 10] = _h[7];
		}
	}
}

// -----------------------------------------------------------------------------------------

#define CHECK_POINT_SEARCH_MODE_MA(_h,incr,mode)  CheckPointSEARCH_MODE_MA(_h,incr,mode,bloomLookUp,BLOOM_BITS,BLOOM_HASHES,maxFound,out)

__device__ __noinline__ void CheckHashCompSEARCH_MODE_MA(uint64_t* px, uint8_t isOdd, int32_t incr,
	uint8_t* bloomLookUp, int BLOOM_BITS, uint8_t BLOOM_HASHES, uint32_t maxFound, uint32_t* out)
{
	uint32_t h[5];
	_GetHash160Comp(px, isOdd, (uint8_t*)h);
	CHECK_POINT_SEARCH_MODE_MA(h, incr, true);
}

// -----------------------------------------------------------------------------------------

#define CHECK_POINT_SEARCH_MODE_SA(_h,incr,mode)  CheckPointSEARCH_MODE_SA(_h,incr,mode,hash160,maxFound,out)

__device__ __noinline__ void CheckHashCompSEARCH_MODE_SA(uint64_t* px, uint8_t isOdd, int32_t incr,
	uint32_t* hash160, uint32_t maxFound, uint32_t* out)
{
	uint32_t h[5];
	_GetHash160Comp(px, isOdd, (uint8_t*)h);
	CHECK_POINT_SEARCH_MODE_SA(h, incr, true);
}
// -----------------------------------------------------------------------------------------

__device__ __noinline__ void CheckHashUnCompSEARCH_MODE_MA(uint64_t* px, uint64_t* py, int32_t incr,
	uint8_t* bloomLookUp, int BLOOM_BITS, uint8_t BLOOM_HASHES, uint32_t maxFound, uint32_t* out)
{
	uint32_t h[5];
	_GetHash160(px, py, (uint8_t*)h);
	CHECK_POINT_SEARCH_MODE_MA(h, incr, false);
}

// ---------------------------------------------------------------------------------------

__device__ __noinline__ void CheckHashUnCompSEARCH_MODE_SA(uint64_t* px, uint64_t* py, int32_t incr,
	uint32_t* hash160, uint32_t maxFound, uint32_t* out)
{
	uint32_t h[5];
	_GetHash160(px, py, (uint8_t*)h);
	CHECK_POINT_SEARCH_MODE_SA(h, incr, false);
}

// -----------------------------------------------------------------------------------------

__device__ __noinline__ void CheckHashSEARCH_MODE_MA(uint32_t mode, uint64_t* px, uint64_t* py, int32_t incr,
	uint8_t* bloomLookUp, int BLOOM_BITS, uint8_t BLOOM_HASHES, uint32_t maxFound, uint32_t* out)
{
	switch (mode) {
	case SEARCH_COMPRESSED:
		CheckHashCompSEARCH_MODE_MA(px, (uint8_t)(py[0] & 1), incr, bloomLookUp, BLOOM_BITS, BLOOM_HASHES, maxFound, out);
		break;
	case SEARCH_UNCOMPRESSED:
		CheckHashUnCompSEARCH_MODE_MA(px, py, incr, bloomLookUp, BLOOM_BITS, BLOOM_HASHES, maxFound, out);
		break;
	case SEARCH_BOTH:
		CheckHashCompSEARCH_MODE_MA(px, (uint8_t)(py[0] & 1), incr, bloomLookUp, BLOOM_BITS, BLOOM_HASHES, maxFound, out);
		CheckHashUnCompSEARCH_MODE_MA(px, py, incr, bloomLookUp, BLOOM_BITS, BLOOM_HASHES, maxFound, out);
		break;
	}
}

// -----------------------------------------------------------------------------------------

#define CHECK_POINT_SEARCH_MODE_MX(_h,incr,mode)  CheckPointSEARCH_MODE_MX(_h,incr,mode,bloomLookUp,BLOOM_BITS,BLOOM_HASHES,maxFound,out)

__device__ __noinline__ void CheckPubCompSEARCH_MODE_MX(uint64_t* px, uint8_t isOdd, int32_t incr,
	uint8_t* bloomLookUp, int BLOOM_BITS, uint8_t BLOOM_HASHES, uint32_t maxFound, uint32_t* out)
{
	uint32_t h[8];
	uint32_t* x32 = (uint32_t*)(px);

	// Compressed public key
	h[0] = __byte_perm(x32[7], 0, 0x0123);
	h[1] = __byte_perm(x32[6], 0, 0x0123);
	h[2] = __byte_perm(x32[5], 0, 0x0123);
	h[3] = __byte_perm(x32[4], 0, 0x0123);
	h[4] = __byte_perm(x32[3], 0, 0x0123);
	h[5] = __byte_perm(x32[2], 0, 0x0123);
	h[6] = __byte_perm(x32[1], 0, 0x0123);
	h[7] = __byte_perm(x32[0], 0, 0x0123);

	CHECK_POINT_SEARCH_MODE_MX(h, incr, true);
}

#define CHECK_POINT_SEARCH_MODE_SX(_h,incr,mode)  CheckPointSEARCH_MODE_SX(_h,incr,mode,xpoint,maxFound,out)

__device__ __noinline__ void CheckPubCompSEARCH_MODE_SX(uint64_t* px, uint8_t isOdd, int32_t incr,
	uint32_t* xpoint, uint32_t maxFound, uint32_t* out)
{
	uint32_t h[8];
	uint32_t* x32 = (uint32_t*)(px);

	// Compressed public key
	h[0] = __byte_perm(x32[7], 0, 0x0123);
	h[1] = __byte_perm(x32[6], 0, 0x0123);
	h[2] = __byte_perm(x32[5], 0, 0x0123);
	h[3] = __byte_perm(x32[4], 0, 0x0123);
	h[4] = __byte_perm(x32[3], 0, 0x0123);
	h[5] = __byte_perm(x32[2], 0, 0x0123);
	h[6] = __byte_perm(x32[1], 0, 0x0123);
	h[7] = __byte_perm(x32[0], 0, 0x0123);

	CHECK_POINT_SEARCH_MODE_SX(h, incr, true);
}

// ---------------------------------------------------------------------------------------

__device__ __noinline__ void CheckPubSEARCH_MODE_MX(uint32_t mode, uint64_t* px, uint64_t* py, int32_t incr,
	uint8_t* bloomLookUp, int BLOOM_BITS, uint8_t BLOOM_HASHES, uint32_t maxFound, uint32_t* out)
{

	if (mode == SEARCH_COMPRESSED) {
		CheckPubCompSEARCH_MODE_MX(px, (uint8_t)(py[0] & 1), incr, bloomLookUp, BLOOM_BITS, BLOOM_HASHES, maxFound, out);
	}
	else {
		return;
	}
}

// -----------------------------------------------------------------------------------------

__device__ __noinline__ void CheckPubSEARCH_MODE_SX(uint32_t mode, uint64_t* px, uint64_t* py, int32_t incr,
	uint32_t* xpoint, uint32_t maxFound, uint32_t* out)
{

	if (mode == SEARCH_COMPRESSED) {
		CheckPubCompSEARCH_MODE_SX(px, (uint8_t)(py[0] & 1), incr, xpoint, maxFound, out);
	}
	else {
		return;
	}
}

// -----------------------------------------------------------------------------------------

#define CHECK_HASH_SEARCH_MODE_MA(incr) CheckHashSEARCH_MODE_MA(mode, px, py, incr, bloomLookUp, BLOOM_BITS, BLOOM_HASHES, maxFound, out)

__device__ void ComputeKeysSEARCH_MODE_MA(uint32_t mode, uint64_t* startx, uint64_t* starty,
	uint8_t* bloomLookUp, int BLOOM_BITS, uint8_t BLOOM_HASHES, uint32_t maxFound, uint32_t* out)
{

	uint64_t dx[GRP_SIZE / 2 + 1][4];
	uint64_t px[4];
	uint64_t py[4];
	uint64_t pyn[4];
	uint64_t sx[4];
	uint64_t sy[4];
	uint64_t dy[4];
	uint64_t _s[4];
	uint64_t _p2[4];

	// Load starting key
	__syncthreads();
	Load256A(sx, startx);
	Load256A(sy, starty);
	Load256(px, sx);
	Load256(py, sy);

	// Fill group with delta x
	uint32_t i;
	for (i = 0; i < HSIZE; i++)
		ModSub256(dx[i], Gx + 4 * i, sx);
	ModSub256(dx[i], Gx + 4 * i, sx);   // For the first point
	ModSub256(dx[i + 1], _2Gnx, sx); // For the next center point

	// Compute modular inverse
	_ModInvGrouped(dx);

	// We use the fact that P + i*G and P - i*G has the same deltax, so the same inverse
	// We compute key in the positive and negative way from the center of the group

	// Check starting point
	CHECK_HASH_SEARCH_MODE_MA(GRP_SIZE / 2);

	ModNeg256(pyn, py);

	for (i = 0; i < HSIZE; i++) {

		// P = StartPoint + i*G
		Load256(px, sx);
		Load256(py, sy);
		ModSub256(dy, Gy + 4 * i, py);

		_ModMult(_s, dy, dx[i]);                 //  s = (p2.y-p1.y)*inverse(p2.x-p1.x)
		_ModSqr(_p2, _s);                        // _p2 = pow2(s)

		ModSub256(px, _p2, px);
		ModSub256(px, Gx + 4 * i);               // px = pow2(s) - p1.x - p2.x;

		ModSub256(py, Gx + 4 * i, px);
		_ModMult(py, _s);                        // py = - s*(ret.x-p2.x)
		ModSub256(py, Gy + 4 * i);               // py = - p2.y - s*(ret.x-p2.x);

		CHECK_HASH_SEARCH_MODE_MA(GRP_SIZE / 2 + (i + 1));

		// P = StartPoint - i*G, if (x,y) = i*G then (x,-y) = -i*G
		Load256(px, sx);
		ModSub256(dy, pyn, Gy + 4 * i);

		_ModMult(_s, dy, dx[i]);                //  s = (p2.y-p1.y)*inverse(p2.x-p1.x)
		_ModSqr(_p2, _s);                       // _p = pow2(s)

		ModSub256(px, _p2, px);
		ModSub256(px, Gx + 4 * i);              // px = pow2(s) - p1.x - p2.x;

		ModSub256(py, px, Gx + 4 * i);
		_ModMult(py, _s);                       // py = s*(ret.x-p2.x)
		ModSub256(py, Gy + 4 * i, py);          // py = - p2.y - s*(ret.x-p2.x);

		CHECK_HASH_SEARCH_MODE_MA(GRP_SIZE / 2 - (i + 1));

	}

	// First point (startP - (GRP_SZIE/2)*G)
	Load256(px, sx);
	Load256(py, sy);
	ModNeg256(dy, Gy + 4 * i);
	ModSub256(dy, py);

	_ModMult(_s, dy, dx[i]);                  //  s = (p2.y-p1.y)*inverse(p2.x-p1.x)
	_ModSqr(_p2, _s);                         // _p = pow2(s)

	ModSub256(px, _p2, px);
	ModSub256(px, Gx + 4 * i);                // px = pow2(s) - p1.x - p2.x;

	ModSub256(py, px, Gx + 4 * i);
	_ModMult(py, _s);                         // py = s*(ret.x-p2.x)
	ModSub256(py, Gy + 4 * i, py);            // py = - p2.y - s*(ret.x-p2.x);

	CHECK_HASH_SEARCH_MODE_MA(0);

	i++;

	// Next start point (startP + GRP_SIZE*G)
	Load256(px, sx);
	Load256(py, sy);
	ModSub256(dy, _2Gny, py);

	_ModMult(_s, dy, dx[i]);              //  s = (p2.y-p1.y)*inverse(p2.x-p1.x)
	_ModSqr(_p2, _s);                     // _p2 = pow2(s)

	ModSub256(px, _p2, px);
	ModSub256(px, _2Gnx);                 // px = pow2(s) - p1.x - p2.x;

	ModSub256(py, _2Gnx, px);
	_ModMult(py, _s);                     // py = - s*(ret.x-p2.x)
	ModSub256(py, _2Gny);                 // py = - p2.y - s*(ret.x-p2.x);


	// Update starting point
	__syncthreads();
	Store256A(startx, px);
	Store256A(starty, py);

}

// -----------------------------------------------------------------------------------------

__device__ __noinline__ void CheckHashSEARCH_MODE_SA(uint32_t mode, uint64_t* px, uint64_t* py, int32_t incr,
	uint32_t* hash160, uint32_t maxFound, uint32_t* out)
{
	switch (mode) {
	case SEARCH_COMPRESSED:
		CheckHashCompSEARCH_MODE_SA(px, (uint8_t)(py[0] & 1), incr, hash160, maxFound, out);
		break;
	case SEARCH_UNCOMPRESSED:
		CheckHashUnCompSEARCH_MODE_SA(px, py, incr, hash160, maxFound, out);
		break;
	case SEARCH_BOTH:
		CheckHashCompSEARCH_MODE_SA(px, (uint8_t)(py[0] & 1), incr, hash160, maxFound, out);
		CheckHashUnCompSEARCH_MODE_SA(px, py, incr, hash160, maxFound, out);
		break;
	}
}

// -----------------------------------------------------------------------------------------

#define CHECK_HASH_SEARCH_MODE_SA(incr) CheckHashSEARCH_MODE_SA(mode, px, py, incr, hash160, maxFound, out)

__device__ void ComputeKeysSEARCH_MODE_SA(uint32_t mode, uint64_t* startx, uint64_t* starty,
	uint32_t* hash160, uint32_t maxFound, uint32_t* out)
{

	uint64_t dx[GRP_SIZE / 2 + 1][4];
	uint64_t px[4];
	uint64_t py[4];
	uint64_t pyn[4];
	uint64_t sx[4];
	uint64_t sy[4];
	uint64_t dy[4];
	uint64_t _s[4];
	uint64_t _p2[4];

	// Load starting key
	__syncthreads();
	Load256A(sx, startx);
	Load256A(sy, starty);
	Load256(px, sx);
	Load256(py, sy);

	// Fill group with delta x
	uint32_t i;
	for (i = 0; i < HSIZE; i++)
		ModSub256(dx[i], Gx + 4 * i, sx);
	ModSub256(dx[i], Gx + 4 * i, sx);   // For the first point
	ModSub256(dx[i + 1], _2Gnx, sx); // For the next center point

	// Compute modular inverse
	_ModInvGrouped(dx);

	// We use the fact that P + i*G and P - i*G has the same deltax, so the same inverse
	// We compute key in the positive and negative way from the center of the group

	// Check starting point
	CHECK_HASH_SEARCH_MODE_SA(GRP_SIZE / 2);

	ModNeg256(pyn, py);

	for (i = 0; i < HSIZE; i++) {

		// P = StartPoint + i*G
		Load256(px, sx);
		Load256(py, sy);
		ModSub256(dy, Gy + 4 * i, py);

		_ModMult(_s, dy, dx[i]);             //  s = (p2.y-p1.y)*inverse(p2.x-p1.x)
		_ModSqr(_p2, _s);                    // _p2 = pow2(s)

		ModSub256(px, _p2, px);
		ModSub256(px, Gx + 4 * i);           // px = pow2(s) - p1.x - p2.x;

		ModSub256(py, Gx + 4 * i, px);
		_ModMult(py, _s);                    // py = - s*(ret.x-p2.x)
		ModSub256(py, Gy + 4 * i);           // py = - p2.y - s*(ret.x-p2.x);

		CHECK_HASH_SEARCH_MODE_SA(GRP_SIZE / 2 + (i + 1));

		// P = StartPoint - i*G, if (x,y) = i*G then (x,-y) = -i*G
		Load256(px, sx);
		ModSub256(dy, pyn, Gy + 4 * i);

		_ModMult(_s, dy, dx[i]);            //  s = (p2.y-p1.y)*inverse(p2.x-p1.x)
		_ModSqr(_p2, _s);                   // _p = pow2(s)

		ModSub256(px, _p2, px);
		ModSub256(px, Gx + 4 * i);          // px = pow2(s) - p1.x - p2.x;

		ModSub256(py, px, Gx + 4 * i);
		_ModMult(py, _s);                   // py = s*(ret.x-p2.x)
		ModSub256(py, Gy + 4 * i, py);      // py = - p2.y - s*(ret.x-p2.x);

		CHECK_HASH_SEARCH_MODE_SA(GRP_SIZE / 2 - (i + 1));

	}

	// First point (startP - (GRP_SZIE/2)*G)
	Load256(px, sx);
	Load256(py, sy);
	ModNeg256(dy, Gy + 4 * i);
	ModSub256(dy, py);

	_ModMult(_s, dy, dx[i]);              //  s = (p2.y-p1.y)*inverse(p2.x-p1.x)
	_ModSqr(_p2, _s);                     // _p = pow2(s)

	ModSub256(px, _p2, px);
	ModSub256(px, Gx + 4 * i);            // px = pow2(s) - p1.x - p2.x;

	ModSub256(py, px, Gx + 4 * i);
	_ModMult(py, _s);                     // py = s*(ret.x-p2.x)
	ModSub256(py, Gy + 4 * i, py);        // py = - p2.y - s*(ret.x-p2.x);

	CHECK_HASH_SEARCH_MODE_SA(0);

	i++;

	// Next start point (startP + GRP_SIZE*G)
	Load256(px, sx);
	Load256(py, sy);
	ModSub256(dy, _2Gny, py);

	_ModMult(_s, dy, dx[i]);             //  s = (p2.y-p1.y)*inverse(p2.x-p1.x)
	_ModSqr(_p2, _s);                    // _p2 = pow2(s)

	ModSub256(px, _p2, px);
	ModSub256(px, _2Gnx);                // px = pow2(s) - p1.x - p2.x;

	ModSub256(py, _2Gnx, px);
	_ModMult(py, _s);                    // py = - s*(ret.x-p2.x)
	ModSub256(py, _2Gny);                // py = - p2.y - s*(ret.x-p2.x);

	// Update starting point
	__syncthreads();
	Store256A(startx, px);
	Store256A(starty, py);

}



// -----------------------------------------------------------------------------------------

#define CHECK_PUB_SEARCH_MODE_MX(incr) CheckPubSEARCH_MODE_MX(mode, px, py, incr, bloomLookUp, BLOOM_BITS, BLOOM_HASHES, maxFound, out)

__device__ void ComputeKeysSEARCH_MODE_MX(uint32_t mode, uint64_t* startx, uint64_t* starty,
	uint8_t* bloomLookUp, int BLOOM_BITS, uint8_t BLOOM_HASHES, uint32_t maxFound, uint32_t* out)
{

	uint64_t dx[GRP_SIZE / 2 + 1][4];
	uint64_t px[4];
	uint64_t py[4];
	uint64_t pyn[4];
	uint64_t sx[4];
	uint64_t sy[4];
	uint64_t dy[4];
	uint64_t _s[4];
	uint64_t _p2[4];

	// Load starting key
	__syncthreads();
	Load256A(sx, startx);
	Load256A(sy, starty);
	Load256(px, sx);
	Load256(py, sy);

	// Fill group with delta x
	uint32_t i;
	for (i = 0; i < HSIZE; i++)
		ModSub256(dx[i], Gx + 4 * i, sx);
	ModSub256(dx[i], Gx + 4 * i, sx);   // For the first point
	ModSub256(dx[i + 1], _2Gnx, sx); // For the next center point

	// Compute modular inverse
	_ModInvGrouped(dx);

	// We use the fact that P + i*G and P - i*G has the same deltax, so the same inverse
	// We compute key in the positive and negative way from the center of the group

	// Check starting point
	CHECK_PUB_SEARCH_MODE_MX(GRP_SIZE / 2);

	ModNeg256(pyn, py);

	for (i = 0; i < HSIZE; i++) {

		// P = StartPoint + i*G
		Load256(px, sx);
		Load256(py, sy);
		ModSub256(dy, Gy + 4 * i, py);

		_ModMult(_s, dy, dx[i]);                 //  s = (p2.y-p1.y)*inverse(p2.x-p1.x)
		_ModSqr(_p2, _s);                        // _p2 = pow2(s)

		ModSub256(px, _p2, px);
		ModSub256(px, Gx + 4 * i);               // px = pow2(s) - p1.x - p2.x;

		ModSub256(py, Gx + 4 * i, px);
		_ModMult(py, _s);                        // py = - s*(ret.x-p2.x)
		ModSub256(py, Gy + 4 * i);               // py = - p2.y - s*(ret.x-p2.x);

		CHECK_PUB_SEARCH_MODE_MX(GRP_SIZE / 2 + (i + 1));

		// P = StartPoint - i*G, if (x,y) = i*G then (x,-y) = -i*G
		Load256(px, sx);
		ModSub256(dy, pyn, Gy + 4 * i);

		_ModMult(_s, dy, dx[i]);                //  s = (p2.y-p1.y)*inverse(p2.x-p1.x)
		_ModSqr(_p2, _s);                       // _p = pow2(s)

		ModSub256(px, _p2, px);
		ModSub256(px, Gx + 4 * i);              // px = pow2(s) - p1.x - p2.x;

		ModSub256(py, px, Gx + 4 * i);
		_ModMult(py, _s);                       // py = s*(ret.x-p2.x)
		ModSub256(py, Gy + 4 * i, py);          // py = - p2.y - s*(ret.x-p2.x);

		CHECK_PUB_SEARCH_MODE_MX(GRP_SIZE / 2 - (i + 1));

	}

	// First point (startP - (GRP_SZIE/2)*G)
	Load256(px, sx);
	Load256(py, sy);
	ModNeg256(dy, Gy + 4 * i);
	ModSub256(dy, py);

	_ModMult(_s, dy, dx[i]);            //  s = (p2.y-p1.y)*inverse(p2.x-p1.x)
	_ModSqr(_p2, _s);                   // _p = pow2(s)

	ModSub256(px, _p2, px);
	ModSub256(px, Gx + 4 * i);         // px = pow2(s) - p1.x - p2.x;

	ModSub256(py, px, Gx + 4 * i);
	_ModMult(py, _s);                  // py = s*(ret.x-p2.x)
	ModSub256(py, Gy + 4 * i, py);     // py = - p2.y - s*(ret.x-p2.x);

	CHECK_PUB_SEARCH_MODE_MX(0);

	i++;

	// Next start point (startP + GRP_SIZE*G)
	Load256(px, sx);
	Load256(py, sy);
	ModSub256(dy, _2Gny, py);

	_ModMult(_s, dy, dx[i]);          //  s = (p2.y-p1.y)*inverse(p2.x-p1.x)
	_ModSqr(_p2, _s);                 // _p2 = pow2(s)

	ModSub256(px, _p2, px);
	ModSub256(px, _2Gnx);             // px = pow2(s) - p1.x - p2.x;

	ModSub256(py, _2Gnx, px);
	_ModMult(py, _s);                 // py = - s*(ret.x-p2.x)
	ModSub256(py, _2Gny);             // py = - p2.y - s*(ret.x-p2.x);

	// Update starting point
	__syncthreads();
	Store256A(startx, px);
	Store256A(starty, py);

}

// -----------------------------------------------------------------------------------------

#define CHECK_PUB_SEARCH_MODE_SX(incr) CheckPubSEARCH_MODE_SX(mode, px, py, incr, xpoint, maxFound, out)

__device__ void ComputeKeysSEARCH_MODE_SX(uint32_t mode, uint64_t* startx, uint64_t* starty,
	uint32_t* xpoint, uint32_t maxFound, uint32_t* out)
{

	uint64_t dx[GRP_SIZE / 2 + 1][4];
	uint64_t px[4];
	uint64_t py[4];
	uint64_t pyn[4];
	uint64_t sx[4];
	uint64_t sy[4];
	uint64_t dy[4];
	uint64_t _s[4];
	uint64_t _p2[4];

	// Load starting key
	__syncthreads();
	Load256A(sx, startx);
	Load256A(sy, starty);
	Load256(px, sx);
	Load256(py, sy);

	// Fill group with delta x
	uint32_t i;
	for (i = 0; i < HSIZE; i++)
		ModSub256(dx[i], Gx + 4 * i, sx);
	ModSub256(dx[i], Gx + 4 * i, sx);      // For the first point
	ModSub256(dx[i + 1], _2Gnx, sx);       // For the next center point

	// Compute modular inverse
	_ModInvGrouped(dx);

	// We use the fact that P + i*G and P - i*G has the same deltax, so the same inverse
	// We compute key in the positive and negative way from the center of the group

	// Check starting point
	CHECK_PUB_SEARCH_MODE_SX(GRP_SIZE / 2);

	ModNeg256(pyn, py);

	for (i = 0; i < HSIZE; i++) {

		// P = StartPoint + i*G
		Load256(px, sx);
		Load256(py, sy);
		ModSub256(dy, Gy + 4 * i, py);

		_ModMult(_s, dy, dx[i]);           //  s = (p2.y-p1.y)*inverse(p2.x-p1.x)
		_ModSqr(_p2, _s);                  // _p2 = pow2(s)

		ModSub256(px, _p2, px);
		ModSub256(px, Gx + 4 * i);         // px = pow2(s) - p1.x - p2.x;

		ModSub256(py, Gx + 4 * i, px);
		_ModMult(py, _s);                  // py = - s*(ret.x-p2.x)
		ModSub256(py, Gy + 4 * i);         // py = - p2.y - s*(ret.x-p2.x);

		CHECK_PUB_SEARCH_MODE_SX(GRP_SIZE / 2 + (i + 1));

		// P = StartPoint - i*G, if (x,y) = i*G then (x,-y) = -i*G
		Load256(px, sx);
		ModSub256(dy, pyn, Gy + 4 * i);

		_ModMult(_s, dy, dx[i]);            //  s = (p2.y-p1.y)*inverse(p2.x-p1.x)
		_ModSqr(_p2, _s);                   // _p = pow2(s)

		ModSub256(px, _p2, px);
		ModSub256(px, Gx + 4 * i);         // px = pow2(s) - p1.x - p2.x;

		ModSub256(py, px, Gx + 4 * i);
		_ModMult(py, _s);                  // py = s*(ret.x-p2.x)
		ModSub256(py, Gy + 4 * i, py);     // py = - p2.y - s*(ret.x-p2.x);

		CHECK_PUB_SEARCH_MODE_SX(GRP_SIZE / 2 - (i + 1));

	}

	// First point (startP - (GRP_SZIE/2)*G)
	Load256(px, sx);
	Load256(py, sy);
	ModNeg256(dy, Gy + 4 * i);
	ModSub256(dy, py);

	_ModMult(_s, dy, dx[i]);           //  s = (p2.y-p1.y)*inverse(p2.x-p1.x)
	_ModSqr(_p2, _s);                  // _p = pow2(s)

	ModSub256(px, _p2, px);
	ModSub256(px, Gx + 4 * i);         // px = pow2(s) - p1.x - p2.x;

	ModSub256(py, px, Gx + 4 * i);
	_ModMult(py, _s);                  // py = s*(ret.x-p2.x)
	ModSub256(py, Gy + 4 * i, py);     // py = - p2.y - s*(ret.x-p2.x);

	CHECK_PUB_SEARCH_MODE_SX(0);

	i++;

	// Next start point (startP + GRP_SIZE*G)
	Load256(px, sx);
	Load256(py, sy);
	ModSub256(dy, _2Gny, py);

	_ModMult(_s, dy, dx[i]);           //  s = (p2.y-p1.y)*inverse(p2.x-p1.x)
	_ModSqr(_p2, _s);                  // _p2 = pow2(s)

	ModSub256(px, _p2, px);
	ModSub256(px, _2Gnx);              // px = pow2(s) - p1.x - p2.x;

	ModSub256(py, _2Gnx, px);
	_ModMult(py, _s);                  // py = - s*(ret.x-p2.x)
	ModSub256(py, _2Gny);              // py = - p2.y - s*(ret.x-p2.x);

	// Update starting point
	__syncthreads();
	Store256A(startx, px);
	Store256A(starty, py);

}

// ------------------------------------------------------------------------------------------------------------
// ------------------------------------------------------------------------------------------------------------
// -----------------------------------------------------------------------------------------
// -----------------------------------------------------------------------------------------


__device__ __noinline__ void CheckPointSEARCH_ETH_MODE_MA(uint32_t* _h, int32_t incr,
	uint8_t* bloomLookUp, uint64_t BLOOM_BITS, uint8_t BLOOM_HASHES, uint32_t maxFound, uint32_t* out)
{
	uint32_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (BloomCheck(_h, bloomLookUp, BLOOM_BITS, BLOOM_HASHES, 20) > 0) {
		uint32_t pos = atomicAdd(out, 1);
		if (pos < maxFound) {
			out[pos * ITEM_SIZE_A32 + 1] = tid;
			out[pos * ITEM_SIZE_A32 + 2] = (uint32_t)(incr << 16);// | (uint32_t)(mode << 15);// | (uint32_t)(endo);
			out[pos * ITEM_SIZE_A32 + 3] = _h[0];
			out[pos * ITEM_SIZE_A32 + 4] = _h[1];
			out[pos * ITEM_SIZE_A32 + 5] = _h[2];
			out[pos * ITEM_SIZE_A32 + 6] = _h[3];
			out[pos * ITEM_SIZE_A32 + 7] = _h[4];
		}
	}
}


#define CHECK_POINT_SEARCH_ETH_MODE_MA(_h,incr)  CheckPointSEARCH_ETH_MODE_MA(_h,incr,bloomLookUp,BLOOM_BITS,BLOOM_HASHES,maxFound,out)

__device__ __noinline__ void CheckHashCompSEARCH_ETH_MODE_MA(uint64_t* px, uint64_t* py, int32_t incr,
	uint8_t* bloomLookUp, int BLOOM_BITS, uint8_t BLOOM_HASHES, uint32_t maxFound, uint32_t* out)
{
	uint32_t h[5];
	_GetHashKeccak160(px, py, h);
	CHECK_POINT_SEARCH_ETH_MODE_MA(h, incr);
}


__device__ __noinline__ void CheckHashSEARCH_ETH_MODE_MA(uint64_t* px, uint64_t* py, int32_t incr,
	uint8_t* bloomLookUp, int BLOOM_BITS, uint8_t BLOOM_HASHES, uint32_t maxFound, uint32_t* out)
{
	CheckHashCompSEARCH_ETH_MODE_MA(px, py, incr, bloomLookUp, BLOOM_BITS, BLOOM_HASHES, maxFound, out);

}

#define CHECK_HASH_SEARCH_ETH_MODE_MA(incr) CheckHashSEARCH_ETH_MODE_MA(px, py, incr, bloomLookUp, BLOOM_BITS, BLOOM_HASHES, maxFound, out)

__device__ void ComputeKeysSEARCH_ETH_MODE_MA(uint64_t* startx, uint64_t* starty,
	uint8_t* bloomLookUp, int BLOOM_BITS, uint8_t BLOOM_HASHES, uint32_t maxFound, uint32_t* out)
{

	uint64_t dx[GRP_SIZE / 2 + 1][4];
	uint64_t px[4];
	uint64_t py[4];
	uint64_t pyn[4];
	uint64_t sx[4];
	uint64_t sy[4];
	uint64_t dy[4];
	uint64_t _s[4];
	uint64_t _p2[4];

	// Load starting key
	__syncthreads();
	Load256A(sx, startx);
	Load256A(sy, starty);
	Load256(px, sx);
	Load256(py, sy);

	// Fill group with delta x
	uint32_t i;
	for (i = 0; i < HSIZE; i++)
		ModSub256(dx[i], Gx + 4 * i, sx);
	ModSub256(dx[i], Gx + 4 * i, sx);   // For the first point
	ModSub256(dx[i + 1], _2Gnx, sx); // For the next center point

	// Compute modular inverse
	_ModInvGrouped(dx);

	// We use the fact that P + i*G and P - i*G has the same deltax, so the same inverse
	// We compute key in the positive and negative way from the center of the group

	// Check starting point
	CHECK_HASH_SEARCH_ETH_MODE_MA(GRP_SIZE / 2);

	ModNeg256(pyn, py);

	for (i = 0; i < HSIZE; i++) {

		// P = StartPoint + i*G
		Load256(px, sx);
		Load256(py, sy);
		ModSub256(dy, Gy + 4 * i, py);

		_ModMult(_s, dy, dx[i]);                 //  s = (p2.y-p1.y)*inverse(p2.x-p1.x)
		_ModSqr(_p2, _s);                        // _p2 = pow2(s)

		ModSub256(px, _p2, px);
		ModSub256(px, Gx + 4 * i);               // px = pow2(s) - p1.x - p2.x;

		ModSub256(py, Gx + 4 * i, px);
		_ModMult(py, _s);                        // py = - s*(ret.x-p2.x)
		ModSub256(py, Gy + 4 * i);               // py = - p2.y - s*(ret.x-p2.x);

		CHECK_HASH_SEARCH_ETH_MODE_MA(GRP_SIZE / 2 + (i + 1));

		// P = StartPoint - i*G, if (x,y) = i*G then (x,-y) = -i*G
		Load256(px, sx);
		ModSub256(dy, pyn, Gy + 4 * i);

		_ModMult(_s, dy, dx[i]);                //  s = (p2.y-p1.y)*inverse(p2.x-p1.x)
		_ModSqr(_p2, _s);                       // _p = pow2(s)

		ModSub256(px, _p2, px);
		ModSub256(px, Gx + 4 * i);              // px = pow2(s) - p1.x - p2.x;

		ModSub256(py, px, Gx + 4 * i);
		_ModMult(py, _s);                       // py = s*(ret.x-p2.x)
		ModSub256(py, Gy + 4 * i, py);          // py = - p2.y - s*(ret.x-p2.x);

		CHECK_HASH_SEARCH_ETH_MODE_MA(GRP_SIZE / 2 - (i + 1));

	}

	// First point (startP - (GRP_SZIE/2)*G)
	Load256(px, sx);
	Load256(py, sy);
	ModNeg256(dy, Gy + 4 * i);
	ModSub256(dy, py);

	_ModMult(_s, dy, dx[i]);                  //  s = (p2.y-p1.y)*inverse(p2.x-p1.x)
	_ModSqr(_p2, _s);                         // _p = pow2(s)

	ModSub256(px, _p2, px);
	ModSub256(px, Gx + 4 * i);                // px = pow2(s) - p1.x - p2.x;

	ModSub256(py, px, Gx + 4 * i);
	_ModMult(py, _s);                         // py = s*(ret.x-p2.x)
	ModSub256(py, Gy + 4 * i, py);            // py = - p2.y - s*(ret.x-p2.x);

	CHECK_HASH_SEARCH_ETH_MODE_MA(0);

	i++;

	// Next start point (startP + GRP_SIZE*G)
	Load256(px, sx);
	Load256(py, sy);
	ModSub256(dy, _2Gny, py);

	_ModMult(_s, dy, dx[i]);              //  s = (p2.y-p1.y)*inverse(p2.x-p1.x)
	_ModSqr(_p2, _s);                     // _p2 = pow2(s)

	ModSub256(px, _p2, px);
	ModSub256(px, _2Gnx);                 // px = pow2(s) - p1.x - p2.x;

	ModSub256(py, _2Gnx, px);
	_ModMult(py, _s);                     // py = - s*(ret.x-p2.x)
	ModSub256(py, _2Gny);                 // py = - p2.y - s*(ret.x-p2.x);


	// Update starting point
	__syncthreads();
	Store256A(startx, px);
	Store256A(starty, py);

}



__device__ __noinline__ void CheckPointSEARCH_MODE_SA(uint32_t* _h, int32_t incr,
	uint32_t* hash, uint32_t maxFound, uint32_t* out)
{
	uint32_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (MatchHash(_h, hash)) {
		uint32_t pos = atomicAdd(out, 1);
		if (pos < maxFound) {
			out[pos * ITEM_SIZE_A32 + 1] = tid;
			out[pos * ITEM_SIZE_A32 + 2] = (uint32_t)(incr << 16); // | (uint32_t)(mode << 15);// | (uint32_t)(endo);
			out[pos * ITEM_SIZE_A32 + 3] = _h[0];
			out[pos * ITEM_SIZE_A32 + 4] = _h[1];
			out[pos * ITEM_SIZE_A32 + 5] = _h[2];
			out[pos * ITEM_SIZE_A32 + 6] = _h[3];
			out[pos * ITEM_SIZE_A32 + 7] = _h[4];
		}
	}
}

#define CHECK_POINT_SEARCH_ETH_MODE_SA(_h,incr)  CheckPointSEARCH_MODE_SA(_h,incr,hash,maxFound,out)

__device__ __noinline__ void CheckHashCompSEARCH_ETH_MODE_SA(uint64_t* px, uint64_t* py, int32_t incr,
	uint32_t* hash, uint32_t maxFound, uint32_t* out)
{
	uint32_t h[5];
	_GetHashKeccak160(px, py, h);
	CHECK_POINT_SEARCH_ETH_MODE_SA(h, incr);
}

__device__ __noinline__ void CheckHashSEARCH_ETH_MODE_SA(uint64_t* px, uint64_t* py, int32_t incr,
	uint32_t* hash, uint32_t maxFound, uint32_t* out)
{
	CheckHashCompSEARCH_ETH_MODE_SA(px, py, incr, hash, maxFound, out);

}
#define CHECK_HASH_SEARCH_ETH_MODE_SA(incr) CheckHashSEARCH_ETH_MODE_SA(px, py, incr, hash, maxFound, out)

__device__ void ComputeKeysSEARCH_ETH_MODE_SA(uint64_t* startx, uint64_t* starty,
	uint32_t* hash, uint32_t maxFound, uint32_t* out)
{

	uint64_t dx[GRP_SIZE / 2 + 1][4];
	uint64_t px[4];
	uint64_t py[4];
	uint64_t pyn[4];
	uint64_t sx[4];
	uint64_t sy[4];
	uint64_t dy[4];
	uint64_t _s[4];
	uint64_t _p2[4];

	// Load starting key
	__syncthreads();
	Load256A(sx, startx);
	Load256A(sy, starty);
	Load256(px, sx);
	Load256(py, sy);

	// Fill group with delta x
	uint32_t i;
	for (i = 0; i < HSIZE; i++)
		ModSub256(dx[i], Gx + 4 * i, sx);
	ModSub256(dx[i], Gx + 4 * i, sx);   // For the first point
	ModSub256(dx[i + 1], _2Gnx, sx); // For the next center point

	// Compute modular inverse
	_ModInvGrouped(dx);

	// We use the fact that P + i*G and P - i*G has the same deltax, so the same inverse
	// We compute key in the positive and negative way from the center of the group

	// Check starting point
	CHECK_HASH_SEARCH_ETH_MODE_SA(GRP_SIZE / 2);

	ModNeg256(pyn, py);

	for (i = 0; i < HSIZE; i++) {

		// P = StartPoint + i*G
		Load256(px, sx);
		Load256(py, sy);
		ModSub256(dy, Gy + 4 * i, py);

		_ModMult(_s, dy, dx[i]);             //  s = (p2.y-p1.y)*inverse(p2.x-p1.x)
		_ModSqr(_p2, _s);                    // _p2 = pow2(s)

		ModSub256(px, _p2, px);
		ModSub256(px, Gx + 4 * i);           // px = pow2(s) - p1.x - p2.x;

		ModSub256(py, Gx + 4 * i, px);
		_ModMult(py, _s);                    // py = - s*(ret.x-p2.x)
		ModSub256(py, Gy + 4 * i);           // py = - p2.y - s*(ret.x-p2.x);

		CHECK_HASH_SEARCH_ETH_MODE_SA(GRP_SIZE / 2 + (i + 1));

		// P = StartPoint - i*G, if (x,y) = i*G then (x,-y) = -i*G
		Load256(px, sx);
		ModSub256(dy, pyn, Gy + 4 * i);

		_ModMult(_s, dy, dx[i]);            //  s = (p2.y-p1.y)*inverse(p2.x-p1.x)
		_ModSqr(_p2, _s);                   // _p = pow2(s)

		ModSub256(px, _p2, px);
		ModSub256(px, Gx + 4 * i);          // px = pow2(s) - p1.x - p2.x;

		ModSub256(py, px, Gx + 4 * i);
		_ModMult(py, _s);                   // py = s*(ret.x-p2.x)
		ModSub256(py, Gy + 4 * i, py);      // py = - p2.y - s*(ret.x-p2.x);

		CHECK_HASH_SEARCH_ETH_MODE_SA(GRP_SIZE / 2 - (i + 1));

	}

	// First point (startP - (GRP_SZIE/2)*G)
	Load256(px, sx);
	Load256(py, sy);
	ModNeg256(dy, Gy + 4 * i);
	ModSub256(dy, py);

	_ModMult(_s, dy, dx[i]);              //  s = (p2.y-p1.y)*inverse(p2.x-p1.x)
	_ModSqr(_p2, _s);                     // _p = pow2(s)

	ModSub256(px, _p2, px);
	ModSub256(px, Gx + 4 * i);            // px = pow2(s) - p1.x - p2.x;

	ModSub256(py, px, Gx + 4 * i);
	_ModMult(py, _s);                     // py = s*(ret.x-p2.x)
	ModSub256(py, Gy + 4 * i, py);        // py = - p2.y - s*(ret.x-p2.x);

	CHECK_HASH_SEARCH_ETH_MODE_SA(0);

	i++;

	// Next start point (startP + GRP_SIZE*G)
	Load256(px, sx);
	Load256(py, sy);
	ModSub256(dy, _2Gny, py);

	_ModMult(_s, dy, dx[i]);             //  s = (p2.y-p1.y)*inverse(p2.x-p1.x)
	_ModSqr(_p2, _s);                    // _p2 = pow2(s)

	ModSub256(px, _p2, px);
	ModSub256(px, _2Gnx);                // px = pow2(s) - p1.x - p2.x;

	ModSub256(py, _2Gnx, px);
	_ModMult(py, _s);                    // py = - s*(ret.x-p2.x)
	ModSub256(py, _2Gny);                // py = - p2.y - s*(ret.x-p2.x);

	// Update starting point
	__syncthreads();
	Store256A(startx, px);
	Store256A(starty, py);

}


