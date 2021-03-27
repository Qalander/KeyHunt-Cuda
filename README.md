# KeyHunt-Cuda 
_Hunt for Bitcoin private keys._

This is a modified version of VanitySearch by [JeanLucPons](https://github.com/JeanLucPons/VanitySearch/).

To convert Bitcoin legacy addresses to RIPEMD160 hasehs use this [b58dec](https://github.com/kanhavishva/b58dec).

It is important to binary sort the RIPEMD160 file before giving it to the program, otherwise binary search function would not work properly. To do this work use this [RMD160-Sort](https://github.com/kanhavishva/RMD160-Sort).

A lot of gratitude to all the developers whose codes has been used here.

## Changes

- Renamed from VanitySearch to KeyHunt (inspired from [keyhunt](https://github.com/albertobsd/keyhunt) by albertobsd).
- It searches for RIPEMD160 hashes of the addresses in the given file.
- It supports both CPU and Cuda devices as the original VanitySearch does.
- It uses bloom filter for huge addresses matching, but as you know bloom filter gives false-positive results so we need to verify these results with actual binary data. To verifying bloom results it uses binary search function from [keyhunt](https://github.com/albertobsd/keyhunt) by albertobsd.
- Because binary search requires the whole RIPEMD160 file in the memory so it keeps this file data in the system memory and transfers bloom data only to GPU memory, and binary checking for false-positive results is done on CPU. This way we can load a very large RIPEMD160 binary file.
- Search compressed and un-compressed address only.
- For args parsing it uses [argparse](https://github.com/jamolnng/argparse) by jamolnng)

## ToDo

- Add feature to search in given key-space range.

# Usage
```
Usage: KeyHunt-Cuda [options...]
Options:
    -v, --version          Print version
    -c, --check            Check the workings of the codes
    -u, --uncomp           Search uncompressed addresses
    -b, --both             Search both uncompressed or compressed addresses
    -g, --gpu              Enable GPU calculation
    -i, --gpui             GPU ids: 0,1...: List of GPU(s) to use, default is 0
    -x, --gpux             GPU gridsize: g0x,g0y,g1x,g1y, ...: Specify GPU(s) kernel gridsize, default is 8*(MP number),128
    -o, --out              Outputfile: Output results to the specified file, default: Found.txt
    -m, --max              Specify maximun number of addresses found by each kernel call
    -s, --seed             Seed: Specify a seed for the base key, default is random
    -t, --thread           threadNumber: Specify number of CPU thread, default is number of core
    -e, --nosse            Disable SSE hash function
    -l, --list             List cuda enabled devices
    -r, --rkey             Rkey: Rekey interval in MegaKey, default is disabled
    -n, --nbit             Number of base key random bits
    -f, --file             RIPEMD160 binary hash file path
    -h, --help             Shows this page

```

```
KeyHunt-Cuda.exe -t 0 -g -i 0 -x 256,128 -n 120 -f "G:\BTCADDRESSES\address1-160-sorted.bin"

KeyHunt-Cuda v1.03

MODE         : COMPRESSED
DEVICE       : GPU
CPU THREAD   : 0
GPU IDS      : 0
GPU GRIDSIZE : 256x128
SSE          : YES
SEED         :
RKEY(Mk)     : 0
NBIT         : 120
MAX FOUND    : 65536
HASH160 FILE : G:\BTCADDRESSES\address1-160-sorted.bin
OUTPUT FILE  : Found.txt

Loading      : 100 %
Loaded       : 73,446 address

Bloom at 00000233F3AFE6F0
  Version    : 2.1
  Entries    : 146892
  Error      : 0.0000010000
  Bits       : 4223905
  Bits/Elem  : 28.755175
  Bytes      : 527989 (0 MB)
  Hash funcs : 20

Start Time   : Sun Mar 28 01:39:10 2021
Base Key     : 0000000000000000000000000000000000B2E2584BCDDD8EF7382DF58863DB4B (120 bit)

GPU          : GPU #0 GeForce GTX 1650 (14x64 cores) Grid(256x128)

[00:00:08] [CPU+GPU: 494.87 Mk/s] [GPU: 494.87 Mk/s] [T: 4,026,531,840] [F: 0]

BYE

```

## Building

- Microsoft Visual Studio Community 2019 
- CUDA version 10.0

## License
KeyHunt-Cuda is licensed under GPLv3.

## Disclaimer
ALL THE CODES, PROGRAM AND INFORMATION ARE FOR EDUCATIONAL PURPOSES ONLY. USE IT AT YOUR OWN RISK. THE DEVELOPER WILL NOT BE RESPONSIBLE FOR ANY LOSS, DAMAGE OR CLAIM ARISING FROM USING THIS PROGRAM.

