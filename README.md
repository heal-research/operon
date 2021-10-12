<p align="left">
    <img src="./docs/_static/logo_mini.png" height="80px" />
</p>

# Modern C++ framework for Symbolic Regression

[![Build Status](https://dev.azure.com/heal-research/HEAL.Operon/_apis/build/status/heal-research.operon?branchName=master)](https://dev.azure.com/heal-research/HEAL.Operon/_build/latest?definitionId=4&branchName=master)
[![Documentation Status](https://readthedocs.org/projects/operongp/badge/?version=latest)](https://operongp.readthedocs.io/en/latest/?badge=latest)

*Operon* is a modern C++ framework for [symbolic regression](https://en.wikipedia.org/wiki/Symbolic_regression) that uses [genetic programming](https://en.wikipedia.org/wiki/Genetic_programming) to explore a hypothesis space of possible mathematical expressions in order to find the best-fitting model for a given [regression target](https://en.wikipedia.org/wiki/Regression_analysis).
Its main purpose is to help develop accurate and interpretable white-box models in the area of [system identification](https://en.wikipedia.org/wiki/System_identification). More in-depth documentation available at https://operongp.readthedocs.io/.

## How does it work?

Broadly speaking, genetic programming (GP) is said to evolve a population of "computer programs" ― [AST](https://en.wikipedia.org/wiki/Abstract_syntax_tree)-like structures encoding behavior for a given problem domain ― following the principles of [natural selection](https://en.wikipedia.org/wiki/Natural_selection). It repeatedly combines random program parts keeping only the best results ― the "fittest". Here, the biological concept of [fitness](https://en.wikipedia.org/wiki/Survival_of_the_fittest) is defined as a measure of a program's ability to solve a certain task.

In symbolic regression, the programs represent mathematical expressions typically encoded as [expression trees](https://en.wikipedia.org/wiki/Binary_expression_tree). Fitness is usually defined as [goodness of fit](https://en.wikipedia.org/wiki/Goodness_of_fit) between the dependent variable and the prediction of a tree-encoded model. Iterative selection of best-scoring models followed by random recombination leads naturally to a self-improving process that is able to uncover patterns in the data:

<p align="center">
    <img src="./docs/_static/evo.gif"  />
</p>

# Build instructions

The project requires CMake and a C++17 compliant compiler. On Windows we recommend building with `MinGW` or with your `WSL` distro. We recommend using the latest versions of `Eigen` and `Ceres`.

### Required dependencies
- [Taskflow](https://taskflow.github.io/)
- [Eigen](http://eigen.tuxfamily.org)
- [{fmt}](https://fmt.dev/latest/index.html)

### Optional dependencies
- [Ceres](http://ceres-solver.org/) required to use the fully-featured solvers for bounds constrained robustified non-linear least squares problems
- [cxxopts](https://github.com/jarro2783/cxxopts) required for the cli app.
- [doctest](https://github.com/onqtam/doctest) required for unit tests.
- [python](https://www.python.org/) and [pybind11](https://github.com/pybind/pybind11) required to build the python bindings.

These libraries are well-known and should be available in your distribution's package repository. They can also be easily managed using [conda](https://docs.conda.io/en/latest/) or [vcpkg](https://github.com/Microsoft/vcpkg).

Additionally, CMake will download the following libraries during the build generation phase: 
- [cpp-sort](https://github.com/Morwenn/cpp-sort)
- [fast-float](https://github.com/fastfloat/fast_float)
- [robin-hood](https://github.com/martinus/robin-hood-hashing)
- [infix-parser](https://github.com/foolnotion/pratt-parser-calculator)
- [nanobench](https://github.com/martinus/nanobench)
- [rapidcsv](https://github.com/d99kris/rapidcsv)
- [span-lite](https://github.com/martinmoene/span-lite)
- [vectorclass](https://github.com/vectorclass/version2)
- [vstat](https://github.com/heal-research/vstat)
- [xxhash](https://github.com/Cyan4973/xxHash)

### Build options
The following options can be passed to CMake:
| Option                      | Description |
|:----------------------------|:------------|
| `-DCERES_TINY_SOLVER=ON` | Use the very small and self-contained tiny solver from the Ceres suite for solving non-linear least squares problem. |
| `-DUSE_SINGLE_PRECISION=ON` | Perform model evaluation using floats (single precision) instead of doubles. Great for reducing runtime, might not be appropriate for all purposes.           |
| `-DUSE_OPENLIBM=ON`         | Link against Julia's openlibm, a high performance mathematical library (recommended to improve consistency across compilers and operating systems).            |
| `-DBUILD_TESTS=ON` | Build the unit tests. |
| `-DBUILD_PYBIND=ON` | Build the Python bindings. |
| `-DUSE_JEMALLOC=ON`         | Link against [jemalloc](http://jemalloc.net/), a general purpose `malloc(3)` implementation that emphasizes fragmentation avoidance and scalable concurrency support (mutually exclusive with `tcmalloc`).           |
| `-DUSE_TCMALLOC=ON`         | Link against [tcmalloc](https://google.github.io/tcmalloc/) (thread-caching malloc), a `malloc(3)` implementation that reduces lock contention for multi-threaded programs (mutually exclusive with `jemalloc`).          |
| `-DUSE_MIMALLOC=ON`         | Link against [mimalloc](https://github.com/microsoft/mimalloc) a compact general purpose `malloc(3)` implementation with excellent performance (mutually exclusive with `jemalloc` or `tcmalloc`).          |

## Windows / VCPKG

- Install [vcpkg](https://github.com/Microsoft/vcpkg) following the instructions from https://github.com/Microsoft/vcpkg
- Install the required dependencies: `vcpkg install <deps>`
- `cd <path/to/operon>`
- `mkdir build && cd build`
- `cmake .. -G"Your Visual Studio Version" -DCMAKE_TOOLCHAIN_FILE=[vcpkg root]\scripts\buildsystems\vcpkg.cmake`
- `cmake --build . --config Release`

## GNU/Linux

- Install the required dependencies
- `mkdir build && cd build`
- `cmake .. -DCMAKE_BUILD_TYPE=Release`. Use `Debug` for a debug build, or use `CC=clang CXX=clang++` to build with a different compiler.
- `make`. Add `VERBOSE=1` to get the full compilation output or `-j` for parallel compilation.

# Usage

* Run `operon-gp --help` to see the usage of the console client. This is the easiest way to just start modeling some data. The program expects a csv input file and assumes that the file has a header.
* The Python script provided under `scripts` wraps the `operon-gp` binary and can be used to run bigger experiments. Data can be provided as `csv` or `json` files containing metadata (see `data` folder for examples). The script will run a grid search over a parameter space defined by the user.
* Several examples (C++ and Python) are available  [here](https://github.com/foolnotion/operon/blob/master/examples)

# Installing the Python bindings

Operon comes with Python bindings as well as a scikit learn estimator. To build the bindings the option `-DBUILD_PYBIND=TRUE` must be passed to CMake. The desired install path can be specified using the `CMAKE_INSTALL_PREFIX` variable (for example, `-DCMAKE_INSTALL_PREFIX=/usr/local/lib/python3.8/site-packages`). If an install prefix is not provided CMake will try to detect the default path as reported by Python.

Then, the Python module and package can be installed with `cmake --install .` or `make install` (with `sudo` if needed).

## Usage

### Sklearn estimator
```python
from operon.sklearn import SymbolicRegressor

reg = SymbolicRegressor()

# usual sklearn stuff
reg.fit(X, y)
```

### Operon library
```python
from operon import Dataset, RSquared, etc.
```

# Publications

If you find _Operon_ useful you can cite our work as:
```
@inproceedings{10.1145/3377929.3398099,
    author = {Burlacu, Bogdan and Kronberger, Gabriel and Kommenda, Michael},
    title = {Operon C++: An Efficient Genetic Programming Framework for Symbolic Regression},
    year = {2020},
    isbn = {9781450371278},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/3377929.3398099},
    doi = {10.1145/3377929.3398099},
    booktitle = {Proceedings of the 2020 Genetic and Evolutionary Computation Conference Companion},
    pages = {1562–1570},
    numpages = {9},
    keywords = {symbolic regression, genetic programming, C++},
    location = {Canc\'{u}n, Mexico},
    series = {GECCO '20}
}
```

_Operon_ was also featured in a recent survey of symbolic regression methods, where it showed good results:

```
@misc{lacava2021contemporary,
      title={Contemporary Symbolic Regression Methods and their Relative Performance}, 
      author={William La Cava and Patryk Orzechowski and Bogdan Burlacu and Fabrício Olivetti de França and Marco Virgolin and Ying Jin and Michael Kommenda and Jason H. Moore},
      year={2021},
      eprint={2107.14351},
      archivePrefix={arXiv},
      primaryClass={cs.NE}
}
```
