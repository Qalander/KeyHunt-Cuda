# KeyHunt-Cuda 
_Hunt for Bitcoin private keys._

This is a modified version of VanitySearch by [JeanLucPons](https://github.com/JeanLucPons/VanitySearch/).

Renamed from VanitySearch to KeyHunt (inspired from [keyhunt](https://github.com/albertobsd/keyhunt) by albertobsd).

A lot of gratitude to all the developers whose codes has been used here.

# Features
- Single address(rmd160 hash) searching with mode ```-m ADDREES```
- Multiple addresses(rmd160 hashes) searching with mode ```-m ADDREESES```
- Single xpoint searching with mode ```-m XPOINT```
- Multiple xpoint searching with mode ```-m XPOINTS```
- For XPoint[s] mode use x point of the public key, without 02 or 03 prefix(64 chars).
- Cuda only.

# Usage
- For multiple addresses or xpoints, file format must be binary with sorted data.
- To convert addresses list(text format) to rmd160 hashes binary file use provided python script ```addresses_to_hash160.py```
- To convert pubkeys list(text format) to xpoints binary file use provided python script ```pubkeys_to_xpoint.py```
- After getting binary files from python scripts, use ```BinSort``` tool provided with KeyHunt-Cuda to sort these binary files.
- Don't use XPoint[s] mode with ```uncompressed``` compression type.
- CPU and GPU can not be used together, because the program divides the whole input range into equal parts for all the threads, so use either CPU or GPU so that the whole range can increment by all the threads with consistency.
- Minimum entries for bloom filter should be >= 2.
- RMD160 hashes or XPoints file must be in binary format with sorted data.

## addresses_to_hash160.py
```
python3 addresses_to_hash160.py addresses_in.txt hash160_out.bin
```

## pubkeys_to_xpoint.py
```
python3 pubkeys_to_xpoint.py pubkeys_in.txt xpoints_out.bin
```

## BinSort
For hash160 ```length``` would be ```20``` and for xpoint ```length``` would be ```32```.
```
BinSort.exe
Usage: BinSort.exe length in_file out_file
```

## KeyHunt-Cuda
```
KeyHunt-Cuda.exe -h
KeyHunt-Cuda [OPTIONS...] [TARGETS]
Where TARGETS is one address/xpont, or multiple hashes/xpoints file

-h, --help                               : Display this message
-c, --check                              : Check the working of the codes
-u, --uncomp                             : Search uncompressed points
-b, --both                               : Search both uncompressed or compressed points
-g, --gpu                                : Enable GPU calculation
--gpui GPU ids: 0,1,...                  : List of GPU(s) to use, default is 0
--gpux GPU gridsize: g0x,g0y,g1x,g1y,... : Specify GPU(s) kernel gridsize, default is 8*(Device MP count),128
-t, --thread N                           : Specify number of CPU thread, default is number of core
-i, --in FILE                            : Read rmd160 hashes or xpoints from FILE, should be in binary format with sorted
-o, --out FILE                           : Write keys to FILE, default: Found.txt
-m, --mode MODE                          : Specify search mode where MODE is
                                               ADDRESS  : for single address
                                               ADDRESSES: for multiple hashes/addresses
                                               XPOINT   : for single xpoint
                                               XPOINTS  : for multiple xpoints
-l, --list                               : List cuda enabled devices
--range KEYSPACE                         : Specify the range:
                                               START:END
                                               START:+COUNT
                                               START
                                               :END
                                               :+COUNT
                                               Where START, END, COUNT are in hex format
-r, --rkey Rkey                          : Random key interval in MegaKeys, default is disabled
-v, --version                            : Show version

```


CPU mode:
```
KeyHunt-Cuda.exe -t 4 --gpui 0 --gpux 256,256 -m addresses --range 1:1fffffffff -i puzzle_1_37_hash160_out_sorted.bin

KeyHunt-Cuda v1.01

COMP MODE    : COMPRESSED
SEARCH MODE  : Multi Address
DEVICE       : CPU
CPU THREAD   : 4
SSE          : YES
RKEY         : 0 Mkeys
MAX FOUND    : 65536
HASH160 FILE : puzzle_1_37_hash160_out_sorted.bin
OUTPUT FILE  : Found.txt

Loaded       : 37 addresses

Bloom at 000001C84206BB70
  Version    : 2.1
  Entries    : 74
  Error      : 0.0000010000
  Bits       : 2127
  Bits/Elem  : 28.755175
  Bytes      : 266 (0 MB)
  Hash funcs : 20

Start Time   : Mon Jul 19 01:09:22 2021
Global start : 1 (1 bit)
Global end   : 1FFFFFFFFF (37 bit)
Global range : 1FFFFFFFFE (37 bit)


=================================================================================
PubAddress: 1BgGZ9tcN4rm9KBzDn7KprQz87SZ26SAMH
Priv (WIF): p2pkh:KwDiBf89QgGbjEhKnhXJuH7LrciVrZi3qYjgd9M7rFU73sVHnoWn
Priv (HEX): 1
PubK (HEX): 0279BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
=================================================================================


=================================================================================
PubAddress: 1CUNEBjYrCn2y1SdiUMohaKUi4wpP326Lb
Priv (WIF): p2pkh:KwDiBf89QgGbjEhKnhXJuH7LrciVrZi3qYjgd9M7rFU74sHUHy8S
Priv (HEX): 3
PubK (HEX): 02F9308A019258C31049344F85F89D5229B531C845836F99B08601F113BCE036F9
=================================================================================
[00:00:02] [CPU+GPU: 5.06 Mk/s] [GPU: 0.00 Mk/s] [C: 0.007474 %] [R: 0] [T: 10,272,768 (24 bit)] [F: 2]

....

```


### GPU mode:
## Multiple addresses mode
```
KeyHunt-Cuda.exe -t 0 -g --gpui 0 --gpux 256,256 -m addresses -o Found.txt --range 1:1fffffffff -i puzzle_1_37_hash160_out_sorted.bin

KeyHunt-Cuda v1.01

COMP MODE    : COMPRESSED
SEARCH MODE  : Multi Address
DEVICE       : GPU
CPU THREAD   : 0
GPU IDS      : 0
GPU GRIDSIZE : 256x256
SSE          : YES
RKEY         : 0 Mkeys
MAX FOUND    : 65536
HASH160 FILE : puzzle_1_37_hash160_out_sorted.bin
OUTPUT FILE  : Found.txt

Loaded       : 37 addresses

Bloom at 000001F0C78AAA40
  Version    : 2.1
  Entries    : 74
  Error      : 0.0000010000
  Bits       : 2127
  Bits/Elem  : 28.755175
  Bytes      : 266 (0 MB)
  Hash funcs : 20

Start Time   : Mon Jul 19 01:02:40 2021
Global start : 1 (1 bit)
Global end   : 1FFFFFFFFF (37 bit)
Global range : 1FFFFFFFFE (37 bit)

GPU          : GPU #0 GeForce GTX 1650 (14x64 cores) Grid(256x256)


=================================================================================
PubAddress: 1PgQVLmst3Z314JrQn5TNiys8Hc38TcXJu
Priv (WIF): p2pkh:KwDiBf89QgGbjEhKnhXJuH7LrciVrZi3qYjgd9M7rFUGxXgtm63M
Priv (HEX): 483
PubK (HEX): 038B05B0603ABD75B0C57489E451F811E1AFE54A8715045CDF4888333F3EBC6E8B
=================================================================================

=================================================================================
PubAddress: 1LeBZP5QCwwgXRtmVUvTVrraqPUokyLHqe
Priv (WIF): p2pkh:KwDiBf89QgGbjEhKnhXJuH7LrciVrZi3qYjgd9M7rFUBTL67V6dE
Priv (HEX): 202
PubK (HEX): 03A7A4C30291AC1DB24B4AB00C442AA832F7794B5A0959BEC6E8D7FEE802289DCD
=================================================================================

=================================================================================
PubAddress: 1CQFwcjw1dwhtkVWBttNLDtqL7ivBonGPV
Priv (WIF): p2pkh:KwDiBf89QgGbjEhKnhXJuH7LrciVrZi3qYjgd9M7rFUB3vfDKcxZ
Priv (HEX): 1D3
PubK (HEX): 0243601D61C836387485E9514AB5C8924DD2CFD466AF34AC95002727E1659D60F7
=================================================================================

....

```

## Single address mode:
```
KeyHunt-Cuda.exe -t 0 -g --gpui 0 --gpux 256,256 -m address --range 400000000:7ffffffff 1PWCx5fovoEaoBowAvF5k91m2Xat9bMgwb

KeyHunt-Cuda v1.01

COMP MODE    : COMPRESSED
SEARCH MODE  : Single Address
DEVICE       : GPU
CPU THREAD   : 0
GPU IDS      : 0
GPU GRIDSIZE : 256x256
SSE          : YES
RKEY         : 0 Mkeys
MAX FOUND    : 65536
ADDRESS      : 1PWCx5fovoEaoBowAvF5k91m2Xat9bMgwb
OUTPUT FILE  : Found.txt

Start Time   : Mon Jul 19 01:08:06 2021
Global start : 400000000 (35 bit)
Global end   : 7FFFFFFFF (35 bit)
Global range : 3FFFFFFFF (34 bit)

GPU          : GPU #0 GeForce GTX 1650 (14x64 cores) Grid(256x256)

[00:00:24] [CPU+GPU: 368.81 Mk/s] [GPU: 368.81 Mk/s] [C: 54.687500 %] [R: 0] [T: 9,395,240,960 (34 bit)] [F: 0]
=================================================================================
PubAddress: 1PWCx5fovoEaoBowAvF5k91m2Xat9bMgwb
Priv (WIF): p2pkh:KwDiBf89QgGbjEhKnhXJuH7LrciVrZi3qYjgd9MP7J9oTbu6KuRr
Priv (HEX): 4AED21170
PubK (HEX): 02F6A8148A62320E149CB15C544FE8A25AB483A0095D2280D03B8A00A7FEADA13D
=================================================================================
[00:00:26] [CPU+GPU: 360.62 Mk/s] [GPU: 360.62 Mk/s] [C: 58.593750 %] [R: 0] [T: 10,066,329,600 (34 bit)] [F: 1]

BYE
```

## Multiple XPoints mode
```
KeyHunt-Cuda.exe -t 0 -g --gpui 0 --gpux 256,256 -m xpoints --range 1:1fffffffff -i xpoints_1_37_out_sorted.bin

KeyHunt-Cuda v1.01

COMP MODE    : COMPRESSED
SEARCH MODE  : Multi X Points
DEVICE       : GPU
CPU THREAD   : 0
GPU IDS      : 0
GPU GRIDSIZE : 256x256
SSE          : YES
RKEY         : 0 Mkeys
MAX FOUND    : 65536
XPOINTS FILE : xpoints_1_37_out_sorted.bin
OUTPUT FILE  : Found.txt

Loaded       : 37 xpoints

Bloom at 00000174E464AAF0
  Version    : 2.1
  Entries    : 74
  Error      : 0.0000010000
  Bits       : 2127
  Bits/Elem  : 28.755175
  Bytes      : 266 (0 MB)
  Hash funcs : 20

Start Time   : Mon Jul 19 01:13:05 2021
Global start : 1 (1 bit)
Global end   : 1FFFFFFFFF (37 bit)
Global range : 1FFFFFFFFE (37 bit)

GPU          : GPU #0 GeForce GTX 1650 (14x64 cores) Grid(256x256)


=================================================================================
PubAddress: 1PgQVLmst3Z314JrQn5TNiys8Hc38TcXJu
Priv (WIF): p2pkh:KwDiBf89QgGbjEhKnhXJuH7LrciVrZi3qYjgd9M7rFUGxXgtm63M
Priv (HEX): 483
PubK (HEX): 038B05B0603ABD75B0C57489E451F811E1AFE54A8715045CDF4888333F3EBC6E8B
=================================================================================

=================================================================================
PubAddress: 1LeBZP5QCwwgXRtmVUvTVrraqPUokyLHqe
Priv (WIF): p2pkh:KwDiBf89QgGbjEhKnhXJuH7LrciVrZi3qYjgd9M7rFUBTL67V6dE
Priv (HEX): 202
PubK (HEX): 03A7A4C30291AC1DB24B4AB00C442AA832F7794B5A0959BEC6E8D7FEE802289DCD
=================================================================================

...

```

## Single XPoint mode
```
KeyHunt-Cuda.exe -t 0 -g --gpui 0 --gpux 256,256 -m xpoint --range 8000000000:ffffffffff a2efa402fd5268400c77c20e574ba86409ededee7c4020e4b9f0edbee53de0d4

KeyHunt-Cuda v1.01

COMP MODE    : COMPRESSED
SEARCH MODE  : Single X Point
DEVICE       : GPU
CPU THREAD   : 0
GPU IDS      : 0
GPU GRIDSIZE : 256x256
SSE          : YES
RKEY         : 0 Mkeys
MAX FOUND    : 65536
XPOINT       : a2efa402fd5268400c77c20e574ba86409ededee7c4020e4b9f0edbee53de0d4
OUTPUT FILE  : Found.txt

Start Time   : Sun Jul 18 23:46:15 2021
Global start : 8000000000 (40 bit)
Global end   : FFFFFFFFFF (40 bit)
Global range : 7FFFFFFFFF (39 bit)

GPU          : GPU #0 GeForce GTX 1650 (14x64 cores) Grid(256x256)

[00:05:07] [CPU+GPU: 1012.98 Mk/s] [GPU: 1012.98 Mk/s] [C: 57.568359 %] [R: 0] [T: 316,485,402,624 (39 bit)] [F: 0]
=================================================================================
PubAddress: 1EeAxcprB2PpCnr34VfZdFrkUWuxyiNEFv
Priv (WIF): p2pkh:KwDiBf89QgGbjEhKnhXJuH7LrciVrZi3qYjgd9aFJuCJDo5F6Jm7
Priv (HEX): E9AE4933D6
PubK (HEX): 03A2EFA402FD5268400C77C20E574BA86409EDEDEE7C4020E4B9F0EDBEE53DE0D4
=================================================================================
[00:05:09] [CPU+GPU: 1006.20 Mk/s] [GPU: 1006.20 Mk/s] [C: 57.910156 %] [R: 0] [T: 318,364,450,816 (39 bit)] [F: 1]

BYE
```

## Building
##### Windows
- Microsoft Visual Studio Community 2019 
- CUDA version 10.0
##### Linux
 - Edit the makefile and set up the appropriate CUDA SDK and compiler paths for nvcc. Or pass them as variables to `make` command.
 - Install libgmp: ```sudo apt install -y libgmp-dev```

    ```make
    CUDA       = /usr/local/cuda-11.0
    CXXCUDA    = /usr/bin/g++
    ```
 - To build CPU-only version (without CUDA support):
    ```sh
    $ make all
    ```
 - To build with CUDA: pass CCAP value according to your GPU compute capability
    ```sh
    $ cd KeyHunt-Cuda
    $ make gpu=1 CCAP=75 all
    ```
#### BinSort
```sh
$ cd BinSort
$ make
```
#### Python scripts
```
python3 -m pip install base58
```

## License
KeyHunt-Cuda is licensed under GPLv3.

## Donation
- BTC: bc1qwngus7cv3z45w3xnsxrwru9t705azg4a0mux0h
- ETH: 0x48bD1aE2B289feDBcCcba0D1591E7f088752bd99

## __Disclaimer__
ALL THE CODES, PROGRAM AND INFORMATION ARE FOR EDUCATIONAL PURPOSES ONLY. USE IT AT YOUR OWN RISK. THE DEVELOPER WILL NOT BE RESPONSIBLE FOR ANY LOSS, DAMAGE OR CLAIM ARISING FROM USING THIS PROGRAM.

