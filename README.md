# KeyHunt-Cuda 
_Hunt for Bitcoin private keys._

This is a modified version of VanitySearch by [JeanLucPons](https://github.com/JeanLucPons/VanitySearch/).

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
    -f, --file             RIPEMD160 Address file path                                                                      (Required)
    -h, --help             Shows this page        

```

```
KeyHunt-Cuda.exe -t 8 -g -f G:/BTCADDRESSES/address6-160-sorted.bin

KeyHunt-Cuda v1.0

MODE         : COMPRESSED
DEVICE       : CPU & GPU
CPU THREAD   : 8
GPU IDS      : 0
GPU GRIDSIZE : -1x128
SSE          : YES
SEED         :
RKEY(Mk)     : 0
MAX FOUND    : 65536
HASH160 FILE : G:/BTCADDRESSES/address6-160-sorted.bin
OUTPUT FILE  : Found.txt

Loading      : 100 %
Loaded       : 195,855,350 address

Bloom at 000001EBD359DBC0
  Version    : 2.1
  Entries    : 391710700
  Error      : 0.0000010000
  Bits       : 11263709779
  Bits/Elem  : 28.755175
  Bytes      : 1407963723 (1342 MB)
  Hash funcs : 20

Start Time   : Fri Mar 19 19:44:25 2021
Base Key     : 2FC7AE69A953E920D4E7CA655FB52386A1516AE8F9101FF7826450397D441C00

GPU          : GPU #0 GeForce GTX 1650 (14x64 cores) Grid(112x128)

[00:00:17] [CPU+GPU: 243.40 Mk/s] [GPU: 231.94 Mk/s] [T: 4,249,135,104] [F: 0]

BYE

```

## Building

- Microsoft Visual Studio Community 2019 
- CUDA version 10.0

## License
KeyHunt-Cuda is licensed under GPLv3.

