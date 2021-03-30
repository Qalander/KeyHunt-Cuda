# KeyHunt-Cuda 
_Hunt for Bitcoin private keys._
## This is a experimental project & right now going through a lot of changes, so bugs and errors can appears.

This is a modified version of VanitySearch by [JeanLucPons](https://github.com/JeanLucPons/VanitySearch/).

To convert Bitcoin legacy addresses to RIPEMD160 hasehs, you can use this [b58dec](https://github.com/kanhavishva/b58dec).

It is important to binary sort the RIPEMD160 file before giving it to the program, otherwise binary search function would not work properly. To do this work you can use this [RMD160-Sort](https://github.com/kanhavishva/RMD160-Sort).

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

- Save current state to work file and continue from it.
- Decrement from end range to start range.
- Stop on range completion.
- More friendly command line arguments.

# Usage

CPU and GPU can not be used together, because right now the program divides the whole input range into equal parts for all the threads, so use either CPU or GPU so that the whole range can increment by all the threads with consistency.

Minimum address should be more than 1000.

```
KeyHunt-Cuda.exe -h
Usage: KeyHunt-Cuda [options...]
Options:
    -v, --version          Print version
    -c, --check            Check the working of the codes
    -u, --uncomp           Search uncompressed addresses
    -b, --both             Search both uncompressed or compressed addresses
    -g, --gpu              Enable GPU calculation
    -i, --gpui             GPU ids: 0,1...: List of GPU(s) to use, default is 0
    -x, --gpux             GPU gridsize: g0x,g0y,g1x,g1y, ...: Specify GPU(s) kernel gridsize, default is 8*(Device MP count),128
    -o, --out              Outputfile: Output results to the specified file, default: Found.txt
    -m, --max              Specify maximun number of addresses found by each kernel call
    -t, --thread           threadNumber: Specify number of CPU thread, default is number of core
    -l, --list             List cuda enabled devices
    -f, --file             Ripemd160 binary hash file path
    -s, --start            Range start in hex
    -e, --end              Range end in hex, if not provided then, endRange would be: startRange + 10000000000000000
    -h, --help             Shows this page

```


CPU mode:
```
KeyHunt-Cuda.exe -t 4 -s 10 -f address1-160-sorted.bin

KeyHunt-Cuda v1.04

MODE         : COMPRESSED
DEVICE       : CPU
CPU THREAD   : 4
GPU IDS      : 0
GPU GRIDSIZE : -1x128 (grid size will be calculated automatically based on multiprocessor number on GPU device)
SSE          : YES
MAX FOUND    : 65536
HASH160 FILE : address1-160-sorted.bin
OUTPUT FILE  : Found.txt

Loading      : 100 %
Loaded       : 73,446 address

Bloom at 000002A37A33D030
  Version    : 2.1
  Entries    : 146892
  Error      : 0.0000010000
  Bits       : 4223905
  Bits/Elem  : 28.755175
  Bytes      : 527989 (0 MB)
  Hash funcs : 20

Start Time   : Tue Mar 30 13:23:12 2021
Global start : 0000000000000000000000000000000000000000000000000000000000000010 (5 bit)
Global end   : 000000000000000000000000000000000000000000000000002386F26FC10010 (54 bit)

CPU Thread 00: 0000000000000000000000000000000000000000000000000000000000000010 : 0000000000000000000000000000000000000000000000000008E1BC9BF04010
CPU Thread 01: 0000000000000000000000000000000000000000000000000008E1BC9BF04010 : 0000000000000000000000000000000000000000000000000011C37937E08010
CPU Thread 02: 0000000000000000000000000000000000000000000000000011C37937E08010 : 000000000000000000000000000000000000000000000000001AA535D3D0C010
             .
CPU Thread 03: 000000000000000000000000000000000000000000000000001AA535D3D0C010 : 000000000000000000000000000000000000000000000000002386F26FC10010

[00:00:06] [CPU+GPU: 7.17 Mk/s] [GPU: 0.00 Mk/s] [T: 43,671,552] [F: 18]

BYE

```


GPU mode:
```
KeyHunt-Cuda.exe -t 0 -g -i 0 -x 256,128 -s 1 -f address1-160-sorted.bin

KeyHunt-Cuda v1.04

MODE         : COMPRESSED
DEVICE       : GPU
CPU THREAD   : 0
GPU IDS      : 0
GPU GRIDSIZE : 256x128
SSE          : YES
MAX FOUND    : 65536
HASH160 FILE : address1-160-sorted.bin
OUTPUT FILE  : Found.txt

Loading      : 100 %
Loaded       : 73,446 address

Bloom at 000001B2EAC6F2C0
  Version    : 2.1
  Entries    : 146892
  Error      : 0.0000010000
  Bits       : 4223905
  Bits/Elem  : 28.755175
  Bytes      : 527989 (0 MB)
  Hash funcs : 20

Start Time   : Tue Mar 30 13:30:40 2021
Global start : 0000000000000000000000000000000000000000000000000000000000000001 (1 bit)
Global end   : 000000000000000000000000000000000000000000000000002386F26FC10001 (54 bit)

GPU          : GPU #0 GeForce GTX 1650 (14x64 cores) Grid(256x128)

GPU 0 Thread 000000: 0000000000000000000000000000000000000000000000000000000000000001 : 000000000000000000000000000000000000000000000000000000470DE4DF83
GPU 0 Thread 000001: 000000000000000000000000000000000000000000000000000000470DE4DF83 : 0000000000000000000000000000000000000000000000000000008E1BC9BF05
GPU 0 Thread 000002: 0000000000000000000000000000000000000000000000000000008E1BC9BF05 : 000000000000000000000000000000000000000000000000000000D529AE9E87
                  .
GPU 0 Thread 032767: 000000000000000000000000000000000000000000000000002386AB61DC207F : 000000000000000000000000000000000000000000000000002386F26FC10001

[00:00:04] [CPU+GPU: 495.66 Mk/s] [GPU: 495.66 Mk/s] [T: 2,013,265,920] [F: 14]

BYE
```

## Building

- Microsoft Visual Studio Community 2019 
- CUDA version 10.0

## License
KeyHunt-Cuda is licensed under GPLv3.

## Disclaimer
ALL THE CODES, PROGRAM AND INFORMATION ARE FOR EDUCATIONAL PURPOSES ONLY. USE IT AT YOUR OWN RISK. THE DEVELOPER WILL NOT BE RESPONSIBLE FOR ANY LOSS, DAMAGE OR CLAIM ARISING FROM USING THIS PROGRAM.

