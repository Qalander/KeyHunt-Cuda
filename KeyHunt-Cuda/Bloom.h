#ifndef BLOOMFILTER_H
#define BLOOMFILTER_H


class Bloom
{
public:
    Bloom(unsigned long long int entries, double error);
    ~Bloom();
    int check(const void *buffer, int len);
    int add(const void *buffer, int len);
    void print();
    int reset();
    int save(const char *filename);
    int load(const char *filename);

    unsigned char get_hashes();
    unsigned long long int get_bits();
    unsigned long long int get_bytes();
    const unsigned char *get_bf();

private:
    static unsigned int murmurhash2(const void *key, int len, const unsigned int seed);
    int test_bit_set_bit(unsigned char *buf, unsigned int bit, int set_bit);
    int bloom_check_add(const void *buffer, int len, int add);

private:
    // These fields are part of the public interface of this structure.
    // Client code may read these values if desired. Client code MUST NOT
    // modify any of these.
    unsigned long long int _entries;
    unsigned long long int _bits;
    unsigned long long int _bytes;
    unsigned char _hashes;
    double _error;

    // Fields below are private to the implementation. These may go away or
    // change incompatibly at any moment. Client code MUST NOT access or rely
    // on these.
    unsigned char _ready;
    unsigned char _major;
    unsigned char _minor;
    double _bpe;
    unsigned char *_bf;
};

#endif // BLOOMFILTER_H
