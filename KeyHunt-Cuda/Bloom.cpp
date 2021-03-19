#include "Bloom.h"
#include <iostream>
#include <math.h>
#include <string.h>
//#include <unistd.h>

#define MAKESTRING(n) STRING(n)
#define STRING(n) #n
#define BLOOM_MAGIC "libbloom2"
#define BLOOM_VERSION_MAJOR 2
#define BLOOM_VERSION_MINOR 1

Bloom::Bloom(unsigned long long entries, double error) : _ready(0)
{
    if (entries < 1000 || error <= 0 || error >= 1) {
        printf("Bloom init error\n");
        return;
    }

    _entries = entries;
    _error = error;

    long double num = -log(_error);
    long double denom = 0.480453013918201; // ln(2)^2
    _bpe = (num / denom);

    long double dentries = (long double)_entries;
    long double allbits = dentries * _bpe;
    _bits = (unsigned long long int)allbits;

    if (_bits % 8) {
        _bytes = (unsigned long long int)(_bits / 8) + 1;
    } else {
        _bytes = (unsigned long long int) _bits / 8;
    }

    _hashes = (unsigned char)ceil(0.693147180559945 * _bpe);  // ln(2)

    _bf = (unsigned char *)calloc((unsigned long long int)_bytes, sizeof(unsigned char));
    if (_bf == NULL) {                                   // LCOV_EXCL_START
        printf("Bloom init error\n");
        return;
    }                                                          // LCOV_EXCL_STOP

    _ready = 1;

    _major = BLOOM_VERSION_MAJOR;
    _minor = BLOOM_VERSION_MINOR;

}
Bloom::~Bloom()
{
    if (_ready)
        free(_bf);
}

int Bloom::check(const void *buffer, int len)
{
    return bloom_check_add(buffer, len, 0);
}


int Bloom::add(const void *buffer, int len)
{
    return bloom_check_add(buffer, len, 1);
}


void Bloom::print()
{
    printf("Bloom at %p\n", (void *)this);
    if (!_ready) {
        printf(" *** NOT READY ***\n");
    }
    printf("  Version    : %d.%d\n", _major, _minor);
    printf("  Entries    : %llu\n", _entries);
    printf("  Error      : %1.10f\n", _error);
    printf("  Bits       : %llu\n", _bits);
    printf("  Bits/Elem  : %f\n", _bpe);
    printf("  Bytes      : %llu", _bytes);
    unsigned int KB = _bytes / 1024;
    unsigned int MB = KB / 1024;
    //printf(" (%u KB, %u MB)\n", KB, MB);
    printf(" (%u MB)\n", MB);
    printf("  Hash funcs : %d\n", _hashes);
}


int Bloom::reset()
{
    if (!_ready)
        return 1;
    memset(_bf, 0, _bytes);
    return 0;
}


int Bloom::save(const char *filename)
{
//    if (filename == NULL || filename[0] == 0) {
//        return 1;
//    }

//    int fd = open(filename, O_WRONLY | O_CREAT, 0644);
//    if (fd < 0) {
//        return 1;
//    }

//    ssize_t out = write(fd, BLOOM_MAGIC, strlen(BLOOM_MAGIC));
//    if (out != strlen(BLOOM_MAGIC)) {
//        goto save_error;        // LCOV_EXCL_LINE
//    }

//    uint16_t size = sizeof(struct bloom);
//    out = write(fd, &size, sizeof(uint16_t));
//    if (out != sizeof(uint16_t)) {
//        goto save_error;        // LCOV_EXCL_LINE
//    }

//    out = write(fd, bloom, sizeof(struct bloom));
//    if (out != sizeof(struct bloom)) {
//        goto save_error;        // LCOV_EXCL_LINE
//    }

//    out = write(fd, _bf, _bytes);
//    if (out != _bytes) {
//        goto save_error;        // LCOV_EXCL_LINE
//    }

//    close(fd);
//    return 0;
//    // LCOV_EXCL_START
//save_error:
//    close(fd);
//    return 1;
//    // LCOV_EXCL_STOP
    return 0;
}


int Bloom::load(const char *filename)
{
//    int rv = 0;

//    if (filename == NULL || filename[0] == 0) {
//        return 1;
//    }
//    if (bloom == NULL) {
//        return 2;
//    }

//    memset(bloom, 0, sizeof(struct bloom));

//    int fd = open(filename, O_RDONLY);
//    if (fd < 0) {
//        return 3;
//    }

//    char line[30];
//    memset(line, 0, 30);
//    ssize_t in = read(fd, line, strlen(BLOOM_MAGIC));

//    if (in != strlen(BLOOM_MAGIC)) {
//        rv = 4;
//        goto load_error;
//    }

//    if (strncmp(line, BLOOM_MAGIC, strlen(BLOOM_MAGIC))) {
//        rv = 5;
//        goto load_error;
//    }

//    uint16_t size;
//    in = read(fd, &size, sizeof(uint16_t));
//    if (in != sizeof(uint16_t)) {
//        rv = 6;
//        goto load_error;
//    }

//    if (size != sizeof(struct bloom)) {
//        rv = 7;
//        goto load_error;
//    }

//    in = read(fd, bloom, sizeof(struct bloom));
//    if (in != sizeof(struct bloom)) {
//        rv = 8;
//        goto load_error;
//    }

//    _bf = NULL;
//    if (_major != BLOOM_VERSION_MAJOR) {
//        rv = 9;
//        goto load_error;
//    }

//    _bf = (unsigned char *)malloc(_bytes);
//    if (_bf == NULL) {
//        rv = 10;        // LCOV_EXCL_LINE
//        goto load_error;
//    }

//    in = read(fd, _bf, _bytes);
//    if (in != _bytes) {
//        rv = 11;
//        free(_bf);
//        _bf = NULL;
//        goto load_error;
//    }

//    close(fd);
//    return rv;

//load_error:
//    close(fd);
//    _ready = 0;
//    return rv;
    return 0;
}


unsigned char Bloom::get_hashes()
{
    return _hashes;
}
unsigned long long int Bloom::get_bits()
{
    return _bits;
}
unsigned long long int Bloom::get_bytes()
{
    return _bytes;
}
const unsigned char *Bloom::get_bf()
{
    return _bf;
}

int Bloom::test_bit_set_bit(unsigned char *buf, unsigned int bit, int set_bit)
{
    unsigned int byte = bit >> 3;
    unsigned char c = buf[byte];        // expensive memory access
    unsigned char mask = 1 << (bit % 8);

    if (c & mask) {
        return 1;
    } else {
        if (set_bit) {
            buf[byte] = c | mask;
        }
        return 0;
    }
}

int Bloom::bloom_check_add(const void *buffer, int len, int add)
{
    if (_ready == 0) {
        printf("bloom not initialized!\n");
        return -1;
    }

    unsigned char hits = 0;
    unsigned int a = murmurhash2(buffer, len, 0x9747b28c);
    unsigned int b = murmurhash2(buffer, len, a);
    unsigned int x;
    unsigned char i;

    for (i = 0; i < _hashes; i++) {
        x = (a + b * i) % _bits;
        if (test_bit_set_bit(_bf, x, add)) {
            hits++;
        } else if (!add) {
            // Don't care about the presence of all the bits. Just our own.
            return 0;
        }
    }

    if (hits == _hashes) {
        return 1;                // 1 == element already in (or collision)
    }

    return 0;
}

// MurmurHash2, by Austin Appleby

// Note - This code makes a few assumptions about how your machine behaves -

// 1. We can read a 4-byte value from any address without crashing
// 2. sizeof(int) == 4

// And it has a few limitations -

// 1. It will not work incrementally.
// 2. It will not produce the same results on little-endian and big-endian
//    machines.
unsigned int Bloom::murmurhash2(const void *key, int len, const unsigned int seed)
{
    // 'm' and 'r' are mixing constants generated offline.
    // They're not really 'magic', they just happen to work well.

    const unsigned int m = 0x5bd1e995;
    const int r = 24;

    // Initialize the hash to a 'random' value

    unsigned int h = seed ^ len;

    // Mix 4 bytes at a time into the hash

    const unsigned char *data = (const unsigned char *)key;

    while (len >= 4) {
        unsigned int k = *(unsigned int *)data;

        k *= m;
        k ^= k >> r;
        k *= m;

        h *= m;
        h ^= k;

        data += 4;
        len -= 4;
    }

    // Handle the last few bytes of the input array

    switch (len) {
    case 3: h ^= data[2] << 16;
    case 2: h ^= data[1] << 8;
    case 1: h ^= data[0];
        h *= m;
    };

    // Do a few final mixes of the hash to ensure the last few
    // bytes are well-incorporated.

    h ^= h >> 13;
    h *= m;
    h ^= h >> 15;

    return h;
}



