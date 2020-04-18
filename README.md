# Introduction 

*Operon* is a [Genetic Programming](https://en.wikipedia.org/wiki/Genetic_programming) (GP) system written in modern C++ with an emphasis on usability and performance.

## Why yet another GP framework? 

*Operon*'s main purpose is to help us test new concepts mainly in the area of [symbolic regression](https://en.wikipedia.org/wiki/Symbolic_regression). That is, we evolve populations of expression trees with the aim of producing accurate *and* interpretable white-box models for solving [system identification](https://en.wikipedia.org/wiki/System_identification) tasks. 

At the same time, we wanted an efficient implementation using modern concepts and idioms like those outlined in the [C++ Core Guidelines](https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#c-core-guidelines). Our design goals in developing *Operon* are:
* Modern build system (we use [Cmake](https://cmake.org/))
* Crossplatform (*Operon* has been tested on Windows and Linux)
* Reliance on modern language features (C++17) and standard library functions and algorithms
* Easily to extend
* Unit test coverage

# Features 

## Encoding
*Operon* uses a linear postfix representation for GP trees. All operators manipulate the representation directly, with no intermediate step. This allows us to leverage a more modern design (no pointers to parent or child nodes) and achieve superior runtime performance.

## Genetic operators
* proportional or tournament selection
* one- and multi-point mutation,
* subtree crossover
* tree initialization with the grow method

## Fitness evaluation
* Support common performance metrics R^2, MSE, NMSE, etc
* Efficient model evaluation using [Eigen](https://eigen.tuxfamily.org/)
* Efficient parallelization using Intel's [thread building blocks](https://github.com/intel/tbb) library
* Support numerical and automatic differentiation of expression trees
* Hybridization with local search
    - Uses a best in class non-linear least squares solver from [Ceres](http://ceres-solver.org/) 
    - Support both *Baldwinian* and *Lamarckian* learning models

## Other algorithmic improvements
* Novel tree hashing algorithm ([paper](https://dblp.org/rec/journals/corr/abs-1902-00882))
* Hash-based tree distance measure enabling fast calculation of population diversity

# Installation

The following dependencies need to be satisfied:
* [Intel-tbb](https://github.com/intel/tbb)
* [Eigen](http://eigen.tuxfamily.org)
* [Ceres](http://ceres-solver.org/)
* [Cxxopts](https://github.com/jarro2783/cxxopts)
* [{fmt}](https://fmt.dev/latest/index.html)
* [Catch2](https://github.com/catchorg/Catch2)
* [microsoft-gsl](https://github.com/microsoft/GSL)

These libraries are well-known and should be available in your distribution's package repository. On Windows they can be easily managed using [vcpkg](https://github.com/Microsoft/vcpkg).

## Build instructions

Building requires a recent version of [cmake](https://cmake.org/) and the latest gcc compiler (currently only gcc-9.1 supports the parallel STL algorithms backed up by Intel-tbb).

The following options can be passed to CMake:
- `-DUSE_JEMALLOC=ON`

[jemalloc](http://jemalloc.net/) is a general purpose `malloc(3)` implementation that emphasizes fragmentation avoidance and scalable concurrency support. Typically improves performance.

- `-DUSE_TCMALLOC=ON`

[TCMalloc](https://google.github.io/tcmalloc/) is a fast, multi-threaded `malloc(3)` implementation from Google. Typically improves performance.

- `-DUSE_SINGLE_PRECISION=ON`

Enable single-precision model evaluation in Operon. Typically results in 2x performance. Empirical testing did not reveal any downside to enabling this option.

### Windows / VCPKG

- Install [vcpkg](https://github.com/Microsoft/vcpkg) following the instructions from https://github.com/Microsoft/vcpkg
- Install the required dependencies: `vcpkg install <deps>`
- `cd <path/to/operon>`
- `mkdir build && cd build`
- `cmake .. -G"Visual Studio 15 2017 Win64" -DCMAKE_TOOLCHAIN_FILE=[vcpkg root]\scripts\buildsystems\vcpkg.cmake`
- `cmake --build . --config Release`
 
### GNU/Linux

- Install the required dependencies
- `mkdir build && cd build`
- `cmake .. -DCMAKE_BUILD_TYPE=Release`. Use `Debug` for a debug build, or use `CC=clang CXX=clang++` to build with a different compiler.
- `make`. You may add `VERBOSE=1` to get the full compilation output or `-j` for parallel compilation.
