#include <iostream>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <cstdlib>

//#define LENGTH 20

static void swap(long LENGTH, uint8_t* a, uint8_t* b)
{
	uint8_t* t = new uint8_t[LENGTH];
	memcpy(t, a, LENGTH);
	memcpy(a, b, LENGTH);
	memcpy(b, t, LENGTH);
	delete[] t;
}

static void heapify(long LENGTH, uint8_t* arr, int64_t n, int64_t i)
{
	int64_t largest = i;
	int64_t l = 2 * i + 1;
	int64_t r = 2 * i + 2;
	if (l < n && memcmp(arr + (l * LENGTH), arr + (largest * LENGTH), LENGTH) > 0)
		largest = l;
	if (r < n && memcmp(arr + (r * LENGTH), arr + (largest * LENGTH), LENGTH) > 0)
		largest = r;
	if (largest != i) {
		swap(LENGTH, arr + (i * LENGTH), arr + (largest * LENGTH));
		heapify(LENGTH, arr, n, largest);
	}
}

static void heapsort(long LENGTH, uint8_t* arr, int64_t n)
{
	int64_t i;
	for (i = n / 2 - 1; i >= 0; i--)
		heapify(LENGTH, arr, n, i);
	for (i = n - 1; i > 0; i--) {
		swap(LENGTH, arr, arr + (i * LENGTH));
		heapify(LENGTH, arr, i, 0);
	}
}


static int64_t partition(long LENGTH, uint8_t* arr, int64_t n)
{
	uint8_t* pivot = new uint8_t[LENGTH];
	int64_t j, i, t, r = (int64_t)n / 2, jaux = -1, iaux = -1, iflag, jflag;
	i = -1;
	memcpy(pivot, arr + (r * LENGTH), LENGTH);
	i = 0;
	j = n - 1;
	do {
		iflag = 1;
		jflag = 1;
		t = memcmp(arr + (i * LENGTH), pivot, LENGTH);
		iflag = (t <= 0);
		while (i < j && iflag) {
			i++;
			t = memcmp(arr + (i * LENGTH), pivot, LENGTH);
			iflag = (t <= 0);
		}
		t = memcmp(arr + (j * LENGTH), pivot, LENGTH);
		jflag = (t > 0);
		while (i < j && jflag) {
			j--;
			t = memcmp(arr + (j * LENGTH), pivot, LENGTH);
			jflag = (t > 0);
		}
		if (i < j) {
			if (i == r) {
				r = j;
			}
			else {
				if (j == r) {
					r = i;
				}
			}

			swap(LENGTH, arr + (i * LENGTH), arr + (j * LENGTH));
			jaux = j;
			iaux = i;
			j--;
			i++;
		}

	} while (j > i);

	delete[] pivot;

	if (jaux != -1 && iaux != -1) {
		if (iflag || jflag) {
			if (iflag) {
				if (r != j)
					swap(LENGTH, arr + (r * LENGTH), arr + ((j)*LENGTH));
				jaux = j;
			}
			if (jflag) {
				if (r != j - 1)
					swap(LENGTH, arr + (r * LENGTH), arr + ((j - 1) * LENGTH));
				jaux = j - 1;
			}
		}
		else {
			if (r != j)
				swap(LENGTH, arr + (r * LENGTH), arr + ((j)*LENGTH));
			jaux = j;
		}
	}
	else {
		if (iflag && jflag) {
			jaux = r;
		}
		else {
			if (iflag) {
				swap(LENGTH, arr + (r * LENGTH), arr + ((j)*LENGTH));
				jaux = j;
			}
		}
	}
	return jaux;
}


static void insertionsort(long LENGTH, uint8_t* arr, int64_t n)
{
	int64_t j, i;
	uint8_t* arrj;
	uint8_t* key = new uint8_t[LENGTH];
	for (i = 1; i < n; i++) {
		j = i - 1;
		memcpy(key, arr + (i * LENGTH), LENGTH);
		arrj = arr + (j * LENGTH);
		while (j >= 0 && memcmp(arrj, key, LENGTH) > 0) {
			memcpy(arr + ((j + 1) * LENGTH), arrj, LENGTH);
			j--;
			if (j >= 0) {
				arrj = arr + (j * LENGTH);
			}
		}
		memcpy(arr + ((j + 1) * LENGTH), key, LENGTH);
	}
	delete[] key;
}

static void introsort(long LENGTH, uint8_t* arr, int64_t depthLimit, int64_t n)
{
	int64_t p;
	if (n > 1) {
		if (n <= 16) {
			insertionsort(LENGTH, arr, n);
		}
		else {
			if (depthLimit == 0) {
				heapsort(LENGTH, arr, n);
			}
			else {
				p = partition(LENGTH, arr, n);
				if (p >= 2) {
					introsort(LENGTH, arr, depthLimit - 1, p);
				}
				if ((n - (p + 1)) >= 2) {
					introsort(LENGTH, arr + ((p + 1) * LENGTH), depthLimit - 1, n - (p + 1));
				}
			}
		}
	}
}

static void sort(long LENGTH, uint8_t* arr, int64_t n)
{
	int64_t depthLimit = ((int64_t)ceil(log(n))) * 2;
	introsort(LENGTH, arr, depthLimit, n);
}

static void write_file(const uint8_t* DATA, int64_t N, int64_t L, const char* filename)
{
	FILE* fd = fopen(filename, "wb");
	if (fd == NULL) {
		printf("Error: not able to open output file: %s\n", filename);
		exit(1);
	}
	fwrite(DATA, 1, N * L, fd);
	fclose(fd);
}

static void sort_file(long LENGTH, const char* in_filename, const char* out_filename)
{
	if (in_filename == NULL || out_filename == NULL) {
		printf("Error: file names are NULL\n");
		exit(1);
	}

	FILE* fd = fopen(in_filename, "rb");
	if (fd == NULL) {
		printf("Error: not able to open input file: %s\n", in_filename);
		exit(1);
	}

	int64_t N = 0;
	int64_t TOTAL = 0;

#ifdef WIN64
	_fseeki64(fd, 0, SEEK_END);
	TOTAL = _ftelli64(fd);
#else
	fseek(fd, 0, SEEK_END);
	TOTAL = ftell(fd);
#endif

	N = TOTAL / LENGTH;
	rewind(fd);

	printf("Total entries: %llu\n", N);

	if (TOTAL >= INT64_MAX - 1) {
		printf("Error: more then INT64_MAX addresses found in the file.\n");
		exit(1);
	}

	uint8_t* DATA = (uint8_t*)malloc(N * LENGTH);
	memset(DATA, 0, N * LENGTH);

	printf("Reading data...\n");
	fread(DATA, 1, N * LENGTH, fd);
	fclose(fd);
	printf("Reading data complete\n");

	printf("Sorting data...\n");
	sort(LENGTH, DATA, N);
	insertionsort(LENGTH, DATA, N);
	printf("Sorting data complete\n");

	printf("Saving data...\n");
	write_file(DATA, N, LENGTH, out_filename);
	printf("Saving data complete\n");

	free(DATA);
}

int main(int argc, const char* argv[])
{
	if (argc != 4) {
		printf("Error: wrong args\n");
		printf("Usage: %s length in_file out_file\n", argv[0]);
		exit(1);
	}
	printf("\n");
	sort_file(::atol(argv[1]), argv[2], argv[3]);

	return 0;
}
