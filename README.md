# KeyHunt-Cuda ## _Brute force Bitcoin private keys._

This is a modified version of VanitySearch by [JeanLucPons](https://github.com/JeanLucPons/VanitySearch/).

## Changes

- Renamed from VanitySearch to KeyHunt (inspired from [keyhunt](https://github.com/albertobsd/keyhunt) by albertobsd).
- It searches for RIPEMD160 hashes of the addresses in the given file.
- It supports both CPU and Cuda devices as the original VanitySearch does.
- It uses bloom filter for huge addresses matching, but as you know bloom filter gives false-positive results so we need to verify these results with actual binary data. To verifying bloom results it uses binary search function from [keyhunt](https://github.com/albertobsd/keyhunt) by albertobsd.
- For args parsing it uses [argparse](https://github.com/jamolnng/argparse) by jamolnng)
## Building

- Microsoft Visual Studio Community 2019 
- CUDA version 10.0

## License
KeyHunt-Cuda is licensed under GPLv3.

