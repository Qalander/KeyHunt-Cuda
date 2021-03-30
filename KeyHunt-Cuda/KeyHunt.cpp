#include "KeyHunt.h"
#include "Base58.h"
#include "Bech32.h"
#include "hash/sha256.h"
#include "hash/sha512.h"
#include "IntGroup.h"
#include "Timer.h"
#include "hash/ripemd160.h"
#include <cstring>
#include <cmath>
#include <algorithm>
#include <iostream>
#ifndef WIN64
#include <pthread.h>
#endif

using namespace std;

Point Gn[CPU_GRP_SIZE / 2];
Point _2Gn;

// ----------------------------------------------------------------------------

KeyHunt::KeyHunt(const std::string& addressFile, int searchMode,
	bool useGpu, const std::string& outputFile, bool useSSE, uint32_t maxFound,
	const std::string& rangeStart, const std::string& rangeEnd, bool& should_exit)
{
	this->searchMode = searchMode;
	this->useGpu = useGpu;
	this->outputFile = outputFile;
	this->useSSE = useSSE;
	this->nbGPUThread = 0;
	this->addressFile = addressFile;
	this->maxFound = maxFound;
	this->searchType = P2PKH;
	this->rangeStart.SetBase16(rangeStart.c_str());
	if (rangeEnd.length() <= 0) {
		this->rangeEnd.Set(&this->rangeStart);
		this->rangeEnd.Add(10000000000000000);
	}
	else {
		this->rangeEnd.SetBase16(rangeEnd.c_str());
		if (!this->rangeEnd.IsGreaterOrEqual(&this->rangeStart)) {
			printf("Start range is bigger than end range, so flipping ranges.\n");
			Int t(this->rangeEnd);
			this->rangeEnd.Set(&this->rangeStart);
			this->rangeStart.Set(&t);
		}
	}
	this->rangeDiff.SetInt32(0);



	secp = new Secp256K1();
	secp->Init();

	// load address file
	uint8_t buf[20];
	FILE* wfd;
	uint64_t N = 0;

	wfd = fopen(this->addressFile.c_str(), "rb");
	if (!wfd) {
		printf("%s can not open\n", this->addressFile.c_str());
		exit(1);
	}

	_fseeki64(wfd, 0, SEEK_END);
	N = _ftelli64(wfd);
	N = N / 20;
	rewind(wfd);

	DATA = (uint8_t*)malloc(N * 20);
	memset(DATA, 0, N * 20);

	bloom = new Bloom(2 * N, 0.000001);

	uint64_t percent = (N - 1) / 100;
	uint64_t i = 0;
	printf("\n");
	while (i < N && !should_exit) {
		memset(buf, 0, 20);
		memset(DATA + (i * 20), 0, 20);
		if (fread(buf, 1, 20, wfd) == 20) {
			bloom->add(buf, 20);
			memcpy(DATA + (i * 20), buf, 20);
			if (i % percent == 0) {
				printf("\rLoading      : %llu %%", (i / percent));
				fflush(stdout);
			}
		}
		i++;
	}
	printf("\n");
	fclose(wfd);

	if (should_exit) {
		delete secp;
		delete bloom;
		if (DATA)
			free(DATA);
		exit(0);
	}

	BLOOM_N = bloom->get_bytes();
	TOTAL_ADDR = N;
	printf("Loaded       : %s address\n", formatThousands(i).c_str());
	printf("\n");

	bloom->print();
	printf("\n");

	// Compute Generator table G[n] = (n+1)*G
	Point g = secp->G;
	Gn[0] = g;
	g = secp->DoubleDirect(g);
	Gn[1] = g;
	for (int i = 2; i < CPU_GRP_SIZE / 2; i++) {
		g = secp->AddDirect(g, secp->G);
		Gn[i] = g;
	}
	// _2Gn = CPU_GRP_SIZE*G
	_2Gn = secp->DoubleDirect(Gn[CPU_GRP_SIZE / 2 - 1]);

	// Constant for endomorphism
	// if a is a nth primitive root of unity, a^-1 is also a nth primitive root.
	// beta^3 = 1 mod p implies also beta^2 = beta^-1 mop (by multiplying both side by beta^-1)
	// (beta^3 = 1 mod p),  beta2 = beta^-1 = beta^2
	// (lambda^3 = 1 mod n), lamba2 = lamba^-1 = lamba^2
	beta.SetBase16("7ae96a2b657c07106e64479eac3434e99cf0497512f58995c1396c28719501ee");
	lambda.SetBase16("5363ad4cc05c30e0a5261c028812645a122e22ea20816678df02967c1b23bd72");
	beta2.SetBase16("851695d49a83f8ef919bb86153cbcb16630fb68aed0a766a3ec693d68e6afa40");
	lambda2.SetBase16("ac9c52b33fa3cf1f5ad9e3fd77ed9ba4a880b9fc8ec739c2e0cfc810b51283ce");

	char* ctimeBuff;
	time_t now = time(NULL);
	ctimeBuff = ctime(&now);
	printf("Start Time   : %s", ctimeBuff);

	printf("Global start : %064s (%d bit)\n", this->rangeStart.GetBase16().c_str(), this->rangeStart.GetBitLength());
	printf("Global end   : %064s (%d bit)\n", this->rangeEnd.GetBase16().c_str(), this->rangeEnd.GetBitLength());

}

KeyHunt::~KeyHunt()
{
	delete secp;
	delete bloom;
	if (DATA)
		free(DATA);
}

// ----------------------------------------------------------------------------

double log1(double x)
{
	// Use taylor series to approximate log(1-x)
	return -x - (x * x) / 2.0 - (x * x * x) / 3.0 - (x * x * x * x) / 4.0;
}

void KeyHunt::output(string addr, string pAddr, string pAddrHex)
{

#ifdef WIN64
	WaitForSingleObject(ghMutex, INFINITE);
#else
	pthread_mutex_lock(&ghMutex);
#endif

	FILE* f = stdout;
	bool needToClose = false;

	if (outputFile.length() > 0) {
		f = fopen(outputFile.c_str(), "a");
		if (f == NULL) {
			printf("Cannot open %s for writing\n", outputFile.c_str());
			f = stdout;
		}
		else {
			needToClose = true;
		}
	}

	if (!needToClose)
		printf("\n");

	fprintf(f, "PubAddress: %s\n", addr.c_str());

	{


		switch (searchType) {
		case P2PKH:
			fprintf(f, "Priv (WIF): p2pkh:%s\n", pAddr.c_str());
			break;
		case P2SH:
			fprintf(f, "Priv (WIF): p2wpkh-p2sh:%s\n", pAddr.c_str());
			break;
		case BECH32:
			fprintf(f, "Priv (WIF): p2wpkh:%s\n", pAddr.c_str());
			break;
		}
		fprintf(f, "Priv (HEX): 0x%s\n", pAddrHex.c_str());

	}

	fprintf(f, "==================================================================\n");

	if (needToClose)
		fclose(f);

#ifdef WIN64
	ReleaseMutex(ghMutex);
#else
	pthread_mutex_unlock(&ghMutex);
#endif

}

// ----------------------------------------------------------------------------

bool KeyHunt::checkPrivKey(string addr, Int& key, int32_t incr, int endomorphism, bool mode)
{

	Int k(&key);

	if (incr < 0) {
		k.Add((uint64_t)(-incr));
		k.Neg();
		k.Add(&secp->order);
	}
	else {
		k.Add((uint64_t)incr);
	}

	// Endomorphisms
	switch (endomorphism) {
	case 1:
		k.ModMulK1order(&lambda);
		break;
	case 2:
		k.ModMulK1order(&lambda2);
		break;
	}

	// Check addresses
	Point p = secp->ComputePublicKey(&k);

	string chkAddr = secp->GetAddress(searchType, mode, p);
	if (chkAddr != addr) {

		//Key may be the opposite one (negative zero or compressed key)
		k.Neg();
		k.Add(&secp->order);
		p = secp->ComputePublicKey(&k);

		string chkAddr = secp->GetAddress(searchType, mode, p);
		if (chkAddr != addr) {
			printf("\nWarning, wrong private key generated !\n");
			printf("  Addr :%s\n", addr.c_str());
			printf("  Check:%s\n", chkAddr.c_str());
			printf("  Endo:%d incr:%d comp:%d\n", endomorphism, incr, mode);
			//return false;
		}

	}

	output(addr, secp->GetPrivAddress(mode, k), k.GetBase16());

	return true;

}

// ----------------------------------------------------------------------------

#ifdef WIN64
DWORD WINAPI _FindKey(LPVOID lpParam)
{
#else
void* _FindKey(void* lpParam)
{
#endif
	TH_PARAM* p = (TH_PARAM*)lpParam;
	p->obj->FindKeyCPU(p);
	return 0;
}

#ifdef WIN64
DWORD WINAPI _FindKeyGPU(LPVOID lpParam)
{
#else
void* _FindKeyGPU(void* lpParam)
{
#endif
	TH_PARAM* p = (TH_PARAM*)lpParam;
	p->obj->FindKeyGPU(p);
	return 0;
}

// ----------------------------------------------------------------------------

void KeyHunt::checkAddresses(bool compressed, Int key, int i, Point p1)
{
	unsigned char h0[20];
	Point pte1[1];
	Point pte2[1];

	// Point
	secp->GetHash160(searchType, compressed, p1, h0);
	if (CheckBloomBinary(h0) > 0) {
		string addr = secp->GetAddress(searchType, compressed, h0);
		if (checkPrivKey(addr, key, i, 0, compressed)) {
			nbFoundKey++;
		}
	}

	// Endomorphism #1
	pte1[0].x.ModMulK1(&p1.x, &beta);
	pte1[0].y.Set(&p1.y);
	secp->GetHash160(searchType, compressed, pte1[0], h0);
	if (CheckBloomBinary(h0) > 0) {
		string addr = secp->GetAddress(searchType, compressed, h0);
		if (checkPrivKey(addr, key, i, 1, compressed)) {
			nbFoundKey++;
		}
	}

	// Endomorphism #2
	pte2[0].x.ModMulK1(&p1.x, &beta2);
	pte2[0].y.Set(&p1.y);
	secp->GetHash160(searchType, compressed, pte2[0], h0);
	if (CheckBloomBinary(h0) > 0) {
		string addr = secp->GetAddress(searchType, compressed, h0);
		if (checkPrivKey(addr, key, i, 2, compressed)) {
			nbFoundKey++;
		}
	}

	// Curve symetrie
	// if (x,y) = k*G, then (x, -y) is -k*G
	p1.y.ModNeg();
	secp->GetHash160(searchType, compressed, p1, h0);
	if (CheckBloomBinary(h0) > 0) {
		string addr = secp->GetAddress(searchType, compressed, h0);
		if (checkPrivKey(addr, key, -i, 0, compressed)) {
			nbFoundKey++;
		}
	}

	// Endomorphism #1
	pte1[0].y.ModNeg();
	secp->GetHash160(searchType, compressed, pte1[0], h0);
	if (CheckBloomBinary(h0) > 0) {
		string addr = secp->GetAddress(searchType, compressed, h0);
		if (checkPrivKey(addr, key, -i, 1, compressed)) {
			nbFoundKey++;
		}
	}

	// Endomorphism #2
	pte2[0].y.ModNeg();
	secp->GetHash160(searchType, compressed, pte2[0], h0);
	if (CheckBloomBinary(h0) > 0) {
		string addr = secp->GetAddress(searchType, compressed, h0);
		if (checkPrivKey(addr, key, -i, 2, compressed)) {
			nbFoundKey++;
		}
	}
}

// ----------------------------------------------------------------------------

void KeyHunt::checkAddressesSSE(bool compressed, Int key, int i, Point p1, Point p2, Point p3, Point p4)
{
	unsigned char h0[20];
	unsigned char h1[20];
	unsigned char h2[20];
	unsigned char h3[20];
	Point pte1[4];
	Point pte2[4];

	// Point -------------------------------------------------------------------------
	secp->GetHash160(searchType, compressed, p1, p2, p3, p4, h0, h1, h2, h3);
	if (CheckBloomBinary(h0) > 0) {
		string addr = secp->GetAddress(searchType, compressed, h0);
		if (checkPrivKey(addr, key, i + 0, 0, compressed)) {
			nbFoundKey++;
		}
	}
	if (CheckBloomBinary(h1) > 0) {
		string addr = secp->GetAddress(searchType, compressed, h1);
		if (checkPrivKey(addr, key, i + 1, 0, compressed)) {
			nbFoundKey++;
		}
	}
	if (CheckBloomBinary(h2) > 0) {
		string addr = secp->GetAddress(searchType, compressed, h2);
		if (checkPrivKey(addr, key, i + 2, 0, compressed)) {
			nbFoundKey++;
		}
	}
	if (CheckBloomBinary(h3) > 0) {
		string addr = secp->GetAddress(searchType, compressed, h3);
		if (checkPrivKey(addr, key, i + 3, 0, compressed)) {
			nbFoundKey++;
		}
	}

	// Endomorphism #1
	// if (x, y) = k * G, then (beta*x, y) = lambda*k*G
	pte1[0].x.ModMulK1(&p1.x, &beta);
	pte1[0].y.Set(&p1.y);
	pte1[1].x.ModMulK1(&p2.x, &beta);
	pte1[1].y.Set(&p2.y);
	pte1[2].x.ModMulK1(&p3.x, &beta);
	pte1[2].y.Set(&p3.y);
	pte1[3].x.ModMulK1(&p4.x, &beta);
	pte1[3].y.Set(&p4.y);

	secp->GetHash160(searchType, compressed, pte1[0], pte1[1], pte1[2], pte1[3], h0, h1, h2, h3);
	if (CheckBloomBinary(h0) > 0) {
		string addr = secp->GetAddress(searchType, compressed, h0);
		if (checkPrivKey(addr, key, i + 0, 1, compressed)) {
			nbFoundKey++;
		}
	}
	if (CheckBloomBinary(h1) > 0) {
		string addr = secp->GetAddress(searchType, compressed, h1);
		if (checkPrivKey(addr, key, i + 1, 1, compressed)) {
			nbFoundKey++;
		}
	}
	if (CheckBloomBinary(h2) > 0) {
		string addr = secp->GetAddress(searchType, compressed, h2);
		if (checkPrivKey(addr, key, i + 2, 1, compressed)) {
			nbFoundKey++;
		}
	}
	if (CheckBloomBinary(h3) > 0) {
		string addr = secp->GetAddress(searchType, compressed, h3);
		if (checkPrivKey(addr, key, i + 3, 1, compressed)) {
			nbFoundKey++;
		}
	}

	// Endomorphism #2
	// if (x, y) = k * G, then (beta2*x, y) = lambda2*k*G
	pte2[0].x.ModMulK1(&p1.x, &beta2);
	pte2[0].y.Set(&p1.y);
	pte2[1].x.ModMulK1(&p2.x, &beta2);
	pte2[1].y.Set(&p2.y);
	pte2[2].x.ModMulK1(&p3.x, &beta2);
	pte2[2].y.Set(&p3.y);
	pte2[3].x.ModMulK1(&p4.x, &beta2);
	pte2[3].y.Set(&p4.y);

	secp->GetHash160(searchType, compressed, pte2[0], pte2[1], pte2[2], pte2[3], h0, h1, h2, h3);
	if (CheckBloomBinary(h0) > 0) {
		string addr = secp->GetAddress(searchType, compressed, h0);
		if (checkPrivKey(addr, key, i + 0, 2, compressed)) {
			nbFoundKey++;
		}
	}
	if (CheckBloomBinary(h1) > 0) {
		string addr = secp->GetAddress(searchType, compressed, h1);
		if (checkPrivKey(addr, key, i + 1, 2, compressed)) {
			nbFoundKey++;
		}
	}
	if (CheckBloomBinary(h2) > 0) {
		string addr = secp->GetAddress(searchType, compressed, h2);
		if (checkPrivKey(addr, key, i + 2, 2, compressed)) {
			nbFoundKey++;
		}
	}
	if (CheckBloomBinary(h3) > 0) {
		string addr = secp->GetAddress(searchType, compressed, h3);
		if (checkPrivKey(addr, key, i + 3, 2, compressed)) {
			nbFoundKey++;
		}
	}

	// Curve symetrie -------------------------------------------------------------------------
	// if (x,y) = k*G, then (x, -y) is -k*G

	p1.y.ModNeg();
	p2.y.ModNeg();
	p3.y.ModNeg();
	p4.y.ModNeg();

	secp->GetHash160(searchType, compressed, p1, p2, p3, p4, h0, h1, h2, h3);
	if (CheckBloomBinary(h0) > 0) {
		string addr = secp->GetAddress(searchType, compressed, h0);
		if (checkPrivKey(addr, key, -(i + 0), 0, compressed)) {
			nbFoundKey++;
		}
	}
	if (CheckBloomBinary(h1) > 0) {
		string addr = secp->GetAddress(searchType, compressed, h1);
		if (checkPrivKey(addr, key, -(i + 1), 0, compressed)) {
			nbFoundKey++;
		}
	}
	if (CheckBloomBinary(h2) > 0) {
		string addr = secp->GetAddress(searchType, compressed, h2);
		if (checkPrivKey(addr, key, -(i + 2), 0, compressed)) {
			nbFoundKey++;
		}
	}
	if (CheckBloomBinary(h3) > 0) {
		string addr = secp->GetAddress(searchType, compressed, h3);
		if (checkPrivKey(addr, key, -(i + 3), 0, compressed)) {
			nbFoundKey++;
		}
	}

	// Endomorphism #1
	// if (x, y) = k * G, then (beta*x, y) = lambda*k*G
	pte1[0].y.ModNeg();
	pte1[1].y.ModNeg();
	pte1[2].y.ModNeg();
	pte1[3].y.ModNeg();

	secp->GetHash160(searchType, compressed, pte1[0], pte1[1], pte1[2], pte1[3], h0, h1, h2, h3);
	if (CheckBloomBinary(h0) > 0) {
		string addr = secp->GetAddress(searchType, compressed, h0);
		if (checkPrivKey(addr, key, -(i + 0), 1, compressed)) {
			nbFoundKey++;
		}
	}
	if (CheckBloomBinary(h1) > 0) {
		string addr = secp->GetAddress(searchType, compressed, h1);
		if (checkPrivKey(addr, key, -(i + 1), 1, compressed)) {
			nbFoundKey++;
		}
	}
	if (CheckBloomBinary(h2) > 0) {
		string addr = secp->GetAddress(searchType, compressed, h2);
		if (checkPrivKey(addr, key, -(i + 2), 1, compressed)) {
			nbFoundKey++;
		}
	}
	if (CheckBloomBinary(h3) > 0) {
		string addr = secp->GetAddress(searchType, compressed, h3);
		if (checkPrivKey(addr, key, -(i + 3), 1, compressed)) {
			nbFoundKey++;
		}
	}

	// Endomorphism #2
	// if (x, y) = k * G, then (beta2*x, y) = lambda2*k*G
	pte2[0].y.ModNeg();
	pte2[1].y.ModNeg();
	pte2[2].y.ModNeg();
	pte2[3].y.ModNeg();

	secp->GetHash160(searchType, compressed, pte2[0], pte2[1], pte2[2], pte2[3], h0, h1, h2, h3);
	if (CheckBloomBinary(h0) > 0) {
		string addr = secp->GetAddress(searchType, compressed, h0);
		if (checkPrivKey(addr, key, -(i + 0), 2, compressed)) {
			nbFoundKey++;
		}
	}
	if (CheckBloomBinary(h1) > 0) {
		string addr = secp->GetAddress(searchType, compressed, h1);
		if (checkPrivKey(addr, key, -(i + 1), 2, compressed)) {
			nbFoundKey++;
		}
	}
	if (CheckBloomBinary(h2) > 0) {
		string addr = secp->GetAddress(searchType, compressed, h2);
		if (checkPrivKey(addr, key, -(i + 2), 2, compressed)) {
			nbFoundKey++;
		}
	}
	if (CheckBloomBinary(h3) > 0) {
		string addr = secp->GetAddress(searchType, compressed, h3);
		if (checkPrivKey(addr, key, -(i + 3), 2, compressed)) {
			nbFoundKey++;
		}
	}
}

// ----------------------------------------------------------------------------
void KeyHunt::getCPUStartingKey(int thId, Int & tRangeStart, Int & key, Point & startP)
{
	key.Set(&tRangeStart);
	Int km(&key);
	km.Add((uint64_t)CPU_GRP_SIZE / 2);
	startP = secp->ComputePublicKey(&km);

}

void KeyHunt::FindKeyCPU(TH_PARAM * ph)
{

	// Global init
	int thId = ph->threadId;
	Int tRangeStart = ph->rangeStart;
	Int tRangeEnd = ph->rangeEnd;
	counters[thId] = 0;

	// CPU Thread
	IntGroup* grp = new IntGroup(CPU_GRP_SIZE / 2 + 1);

	// Group Init
	Int  key;
	Point startP;
	getCPUStartingKey(thId, tRangeStart, key, startP);

	Int dx[CPU_GRP_SIZE / 2 + 1];
	Point pts[CPU_GRP_SIZE];

	Int dy;
	Int dyn;
	Int _s;
	Int _p;
	Point pp;
	Point pn;
	grp->Set(dx);

	ph->hasStarted = true;

	while (!endOfSearch) {

		// Fill group
		int i;
		int hLength = (CPU_GRP_SIZE / 2 - 1);

		for (i = 0; i < hLength; i++) {
			dx[i].ModSub(&Gn[i].x, &startP.x);
		}
		dx[i].ModSub(&Gn[i].x, &startP.x);  // For the first point
		dx[i + 1].ModSub(&_2Gn.x, &startP.x); // For the next center point

		// Grouped ModInv
		grp->ModInv();

		// We use the fact that P + i*G and P - i*G has the same deltax, so the same inverse
		// We compute key in the positive and negative way from the center of the group

		// center point
		pts[CPU_GRP_SIZE / 2] = startP;

		for (i = 0; i < hLength && !endOfSearch; i++) {

			pp = startP;
			pn = startP;

			// P = startP + i*G
			dy.ModSub(&Gn[i].y, &pp.y);

			_s.ModMulK1(&dy, &dx[i]);       // s = (p2.y-p1.y)*inverse(p2.x-p1.x);
			_p.ModSquareK1(&_s);            // _p = pow2(s)

			pp.x.ModNeg();
			pp.x.ModAdd(&_p);
			pp.x.ModSub(&Gn[i].x);           // rx = pow2(s) - p1.x - p2.x;

			pp.y.ModSub(&Gn[i].x, &pp.x);
			pp.y.ModMulK1(&_s);
			pp.y.ModSub(&Gn[i].y);           // ry = - p2.y - s*(ret.x-p2.x);

			// P = startP - i*G  , if (x,y) = i*G then (x,-y) = -i*G
			dyn.Set(&Gn[i].y);
			dyn.ModNeg();
			dyn.ModSub(&pn.y);

			_s.ModMulK1(&dyn, &dx[i]);      // s = (p2.y-p1.y)*inverse(p2.x-p1.x);
			_p.ModSquareK1(&_s);            // _p = pow2(s)

			pn.x.ModNeg();
			pn.x.ModAdd(&_p);
			pn.x.ModSub(&Gn[i].x);          // rx = pow2(s) - p1.x - p2.x;

			pn.y.ModSub(&Gn[i].x, &pn.x);
			pn.y.ModMulK1(&_s);
			pn.y.ModAdd(&Gn[i].y);          // ry = - p2.y - s*(ret.x-p2.x);

			pts[CPU_GRP_SIZE / 2 + (i + 1)] = pp;
			pts[CPU_GRP_SIZE / 2 - (i + 1)] = pn;

		}

		// First point (startP - (GRP_SZIE/2)*G)
		pn = startP;
		dyn.Set(&Gn[i].y);
		dyn.ModNeg();
		dyn.ModSub(&pn.y);

		_s.ModMulK1(&dyn, &dx[i]);
		_p.ModSquareK1(&_s);

		pn.x.ModNeg();
		pn.x.ModAdd(&_p);
		pn.x.ModSub(&Gn[i].x);

		pn.y.ModSub(&Gn[i].x, &pn.x);
		pn.y.ModMulK1(&_s);
		pn.y.ModAdd(&Gn[i].y);

		pts[0] = pn;

		// Next start point (startP + GRP_SIZE*G)
		pp = startP;
		dy.ModSub(&_2Gn.y, &pp.y);

		_s.ModMulK1(&dy, &dx[i + 1]);
		_p.ModSquareK1(&_s);

		pp.x.ModNeg();
		pp.x.ModAdd(&_p);
		pp.x.ModSub(&_2Gn.x);

		pp.y.ModSub(&_2Gn.x, &pp.x);
		pp.y.ModMulK1(&_s);
		pp.y.ModSub(&_2Gn.y);
		startP = pp;

		// Check addresses
		if (useSSE) {

			for (int i = 0; i < CPU_GRP_SIZE && !endOfSearch; i += 4) {

				switch (searchMode) {
				case SEARCH_COMPRESSED:
					checkAddressesSSE(true, key, i, pts[i], pts[i + 1], pts[i + 2], pts[i + 3]);
					break;
				case SEARCH_UNCOMPRESSED:
					checkAddressesSSE(false, key, i, pts[i], pts[i + 1], pts[i + 2], pts[i + 3]);
					break;
				case SEARCH_BOTH:
					checkAddressesSSE(true, key, i, pts[i], pts[i + 1], pts[i + 2], pts[i + 3]);
					checkAddressesSSE(false, key, i, pts[i], pts[i + 1], pts[i + 2], pts[i + 3]);
					break;
				}
			}
		}
		else {

			for (int i = 0; i < CPU_GRP_SIZE && !endOfSearch; i++) {

				switch (searchMode) {
				case SEARCH_COMPRESSED:
					checkAddresses(true, key, i, pts[i]);
					break;
				case SEARCH_UNCOMPRESSED:
					checkAddresses(false, key, i, pts[i]);
					break;
				case SEARCH_BOTH:
					checkAddresses(true, key, i, pts[i]);
					checkAddresses(false, key, i, pts[i]);
					break;
				}
			}
		}

		key.Add((uint64_t)CPU_GRP_SIZE);
		counters[thId] += 6 * CPU_GRP_SIZE; // Point + endo #1 + endo #2 + Symetric point + endo #1 + endo #2
	}
	ph->isRunning = false;
}

// ----------------------------------------------------------------------------

void KeyHunt::getGPUStartingKeys(int thId, Int & tRangeStart, Int & tRangeEnd, int groupSize, int nbThread, Int * keys, Point * p)
{

	Int tRangeDiff(tRangeEnd);
	Int tRangeStart2(tRangeStart);
	Int tRangeEnd2(tRangeStart);

	Int tThreads;
	tThreads.SetInt32(nbThread);
	tRangeDiff.Set(&tRangeEnd);
	tRangeDiff.Sub(&tRangeStart);
	tRangeDiff.Div(&tThreads);

	int rangeShowThreasold = 3;
	int rangeShowCounter = 0;

	for (int i = 0; i < nbThread; i++) {

		keys[i].Set(&tRangeStart2);
		tRangeEnd2.Set(&tRangeStart2);
		tRangeEnd2.Add(&tRangeDiff);


		if (i < rangeShowThreasold) {
			printf("GPU %d Thread %06d: %064s : %064s\n", (thId - 0x80L), i, tRangeStart2.GetBase16().c_str(), tRangeEnd2.GetBase16().c_str());
		}
		else if (rangeShowCounter < 1) {
			printf("                  .\n");
			rangeShowCounter++;
			if (i + 1 == nbThread) {
				printf("GPU %d Thread %06d: %064s : %064s\n", (thId - 0x80L), i, tRangeStart2.GetBase16().c_str(), tRangeEnd2.GetBase16().c_str());
			}
		}
		else if (i + 1 == nbThread) {
			printf("GPU %d Thread %06d: %064s : %064s\n", (thId - 0x80L), i, tRangeStart2.GetBase16().c_str(), tRangeEnd2.GetBase16().c_str());
		}

		tRangeStart2.Add(&tRangeDiff);

		Int k(keys + i);
		// Starting key is at the middle of the group
		k.Add((uint64_t)(groupSize / 2));
		p[i] = secp->ComputePublicKey(&k);
	}
	printf("\n");

}

void KeyHunt::FindKeyGPU(TH_PARAM * ph)
{

	bool ok = true;

#ifdef WITHGPU

	// Global init
	int thId = ph->threadId;
	Int tRangeStart = ph->rangeStart;
	Int tRangeEnd = ph->rangeEnd;

	GPUEngine g(ph->gridSizeX, ph->gridSizeY, ph->gpuId, maxFound, BLOOM_N, bloom->get_bits(), 
		bloom->get_hashes(), bloom->get_bf(), DATA, TOTAL_ADDR);
	int nbThread = g.GetNbThread();
	Point* p = new Point[nbThread];
	Int* keys = new Int[nbThread];
	vector<ITEM> found;

	printf("GPU          : %s\n\n", g.deviceName.c_str());

	counters[thId] = 0;

	g.SetSearchMode(searchMode);
	g.SetSearchType(searchType);

	getGPUStartingKeys(thId, tRangeStart, tRangeEnd, g.GetGroupSize(), nbThread, keys, p);
	ok = g.SetKeys(p);

	ph->hasStarted = true;

	// GPU Thread
	while (ok && !endOfSearch) {

		// Call kernel
		ok = g.Launch(found, false);

		for (int i = 0; i < (int)found.size() && !endOfSearch; i++) {

			ITEM it = found[i];
			//checkAddr(it.hash, keys[it.thId], it.incr, it.endo, it.mode);
			string addr = secp->GetAddress(searchType, it.mode, it.hash);

			if (checkPrivKey(addr, keys[it.thId], it.incr, it.endo, it.mode)) {
				nbFoundKey++;
			}

		}

		if (ok) {
			for (int i = 0; i < nbThread; i++) {
				keys[i].Add((uint64_t)STEP_SIZE);
			}
			counters[thId] += 6ULL * STEP_SIZE * nbThread; // Point +  endo1 + endo2 + symetrics
		}

		//ok = g.ClearOutBuffer();
	}

	delete[] keys;
	delete[] p;

#else
	ph->hasStarted = true;
	printf("GPU code not compiled, use -DWITHGPU when compiling.\n");
#endif

	ph->isRunning = false;

}

// ----------------------------------------------------------------------------

bool KeyHunt::isAlive(TH_PARAM * p)
{

	bool isAlive = true;
	int total = nbCPUThread + nbGPUThread;
	for (int i = 0; i < total; i++)
		isAlive = isAlive && p[i].isRunning;

	return isAlive;

}

// ----------------------------------------------------------------------------

bool KeyHunt::hasStarted(TH_PARAM * p)
{

	bool hasStarted = true;
	int total = nbCPUThread + nbGPUThread;
	for (int i = 0; i < total; i++)
		hasStarted = hasStarted && p[i].hasStarted;

	return hasStarted;

}

// ----------------------------------------------------------------------------

uint64_t KeyHunt::getGPUCount()
{

	uint64_t count = 0;
	for (int i = 0; i < nbGPUThread; i++)
		count += counters[0x80L + i];
	return count;

}

uint64_t KeyHunt::getCPUCount()
{

	uint64_t count = 0;
	for (int i = 0; i < nbCPUThread; i++)
		count += counters[i];
	return count;

}

// ----------------------------------------------------------------------------

void KeyHunt::SetupRanges(uint32_t totalThreads)
{
	Int threads;
	threads.SetInt32(totalThreads);
	rangeDiff.Set(&rangeEnd);
	rangeDiff.Sub(&rangeStart);
	rangeDiff.Div(&threads);
}

// ----------------------------------------------------------------------------

void KeyHunt::Search(int nbThread, std::vector<int> gpuId, std::vector<int> gridSize, bool& should_exit)
{

	double t0;
	double t1;
	endOfSearch = false;
	nbCPUThread = nbThread;
	nbGPUThread = (useGpu ? (int)gpuId.size() : 0);
	nbFoundKey = 0;

	// setup ranges
	SetupRanges(nbCPUThread + nbGPUThread);

	memset(counters, 0, sizeof(counters));

	if (!useGpu)
		printf("\n");

	TH_PARAM* params = (TH_PARAM*)malloc((nbCPUThread + nbGPUThread) * sizeof(TH_PARAM));
	memset(params, 0, (nbCPUThread + nbGPUThread) * sizeof(TH_PARAM));

	int rangeShowThreasold = 3;
	int rangeShowCounter = 0;

	// Launch CPU threads
	for (int i = 0; i < nbCPUThread; i++) {
		params[i].obj = this;
		params[i].threadId = i;
		params[i].isRunning = true;

		params[i].rangeStart.Set(&rangeStart);
		rangeStart.Add(&rangeDiff);
		params[i].rangeEnd.Set(&rangeStart);

		if (i < rangeShowThreasold) {
			printf("CPU Thread %02d: %064s : %064s\n", i, params[i].rangeStart.GetBase16().c_str(), params[i].rangeEnd.GetBase16().c_str());
		}
		else if (rangeShowCounter < 1) {
			printf("             .\n");
			rangeShowCounter++;
			if (i + 1 == nbCPUThread) {
				printf("CPU Thread %02d: %064s : %064s\n", i, params[i].rangeStart.GetBase16().c_str(), params[i].rangeEnd.GetBase16().c_str());
			}
		}
		else if (i + 1 == nbCPUThread) {
			printf("CPU Thread %02d: %064s : %064s\n", i, params[i].rangeStart.GetBase16().c_str(), params[i].rangeEnd.GetBase16().c_str());
		}

#ifdef WIN64
		DWORD thread_id;
		CreateThread(NULL, 0, _FindKey, (void*)(params + i), 0, &thread_id);
		ghMutex = CreateMutex(NULL, FALSE, NULL);
#else
		pthread_t thread_id;
		pthread_create(&thread_id, NULL, &_FindKey, (void*)(params + i));
		ghMutex = PTHREAD_MUTEX_INITIALIZER;
#endif
	}

	// Launch GPU threads
	for (int i = 0; i < nbGPUThread; i++) {
		params[nbCPUThread + i].obj = this;
		params[nbCPUThread + i].threadId = 0x80L + i;
		params[nbCPUThread + i].isRunning = true;
		params[nbCPUThread + i].gpuId = gpuId[i];
		params[nbCPUThread + i].gridSizeX = gridSize[2 * i];
		params[nbCPUThread + i].gridSizeY = gridSize[2 * i + 1];

		params[nbCPUThread + i].rangeStart.Set(&rangeStart);
		rangeStart.Add(&rangeDiff);
		params[nbCPUThread + i].rangeEnd.Set(&rangeStart);


#ifdef WIN64
		DWORD thread_id;
		CreateThread(NULL, 0, _FindKeyGPU, (void*)(params + (nbCPUThread + i)), 0, &thread_id);
#else
		pthread_t thread_id;
		pthread_create(&thread_id, NULL, &_FindKeyGPU, (void*)(params + (nbCPUThread + i)));
#endif
	}

#ifndef WIN64
	setvbuf(stdout, NULL, _IONBF, 0);
#endif
	printf("\n");

	uint64_t lastCount = 0;
	uint64_t gpuCount = 0;
	uint64_t lastGPUCount = 0;

	// Key rate smoothing filter
#define FILTER_SIZE 8
	double lastkeyRate[FILTER_SIZE];
	double lastGpukeyRate[FILTER_SIZE];
	uint32_t filterPos = 0;

	double keyRate = 0.0;
	double gpuKeyRate = 0.0;
	char timeStr[256];

	memset(lastkeyRate, 0, sizeof(lastkeyRate));
	memset(lastGpukeyRate, 0, sizeof(lastkeyRate));

	// Wait that all threads have started
	while (!hasStarted(params)) {
		Timer::SleepMillis(500);
	}

	// Reset timer
	Timer::Init();
	t0 = Timer::get_tick();
	startTime = t0;

	while (isAlive(params)) {

		int delay = 2000;
		while (isAlive(params) && delay > 0) {
			Timer::SleepMillis(500);
			delay -= 500;
		}

		gpuCount = getGPUCount();
		uint64_t count = getCPUCount() + gpuCount;

		t1 = Timer::get_tick();
		keyRate = (double)(count - lastCount) / (t1 - t0);
		gpuKeyRate = (double)(gpuCount - lastGPUCount) / (t1 - t0);
		lastkeyRate[filterPos % FILTER_SIZE] = keyRate;
		lastGpukeyRate[filterPos % FILTER_SIZE] = gpuKeyRate;
		filterPos++;

		// KeyRate smoothing
		double avgKeyRate = 0.0;
		double avgGpuKeyRate = 0.0;
		uint32_t nbSample;
		for (nbSample = 0; (nbSample < FILTER_SIZE) && (nbSample < filterPos); nbSample++) {
			avgKeyRate += lastkeyRate[nbSample];
			avgGpuKeyRate += lastGpukeyRate[nbSample];
		}
		avgKeyRate /= (double)(nbSample);
		avgGpuKeyRate /= (double)(nbSample);

		if (isAlive(params)) {
			memset(timeStr, '\0', 256);
			printf("\r[%s] [CPU+GPU: %.2f Mk/s] [GPU: %.2f Mk/s] [T: %s] [F: %d]  ",
				toTimeStr(t1, timeStr),
				avgKeyRate / 1000000.0,
				avgGpuKeyRate / 1000000.0,
				formatThousands(count).c_str(),
				nbFoundKey);
		}

		lastCount = count;
		lastGPUCount = gpuCount;
		t0 = t1;
		endOfSearch = should_exit;
	}

	free(params);

}

// ----------------------------------------------------------------------------

string KeyHunt::GetHex(vector<unsigned char> &buffer)
{
	string ret;

	char tmp[128];
	for (int i = 0; i < (int)buffer.size(); i++) {
		sprintf(tmp, "%02X", buffer[i]);
		ret.append(tmp);
	}
	return ret;
}

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------

int KeyHunt::CheckBloomBinary(const uint8_t * hash)
{
	if (bloom->check(hash, 20) > 0) {
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
	return 0;
}

std::string KeyHunt::formatThousands(uint64_t x)
{
	char buf[32] = "";

	sprintf(buf, "%llu", x);

	std::string s(buf);

	int len = (int)s.length();

	int numCommas = (len - 1) / 3;

	if (numCommas == 0) {
		return s;
	}

	std::string result = "";

	int count = ((len % 3) == 0) ? 0 : (3 - (len % 3));

	for (int i = 0; i < len; i++) {
		result += s[i];

		if (count++ == 2 && i < len - 1) {
			result += ",";
			count = 0;
		}
	}
	return result;
}

char* KeyHunt::toTimeStr(int sec, char* timeStr)
{
	int h, m, s;
	h = (sec / 3600);
	m = (sec - (3600 * h)) / 60;
	s = (sec - (3600 * h) - (m * 60));
	sprintf(timeStr, "%0*d:%0*d:%0*d", 2, h, 2, m, 2, s);
	return (char*)timeStr;
}


