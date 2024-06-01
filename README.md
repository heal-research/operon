<picture>
    <source media="(prefers-color-scheme: dark)" srcset="./rtd/_static/logo_mini_dark.png">
    <source media="(prefers-color-scheme: light)" srcset="./rtd/_static/logo_mini_light.png">
    <img src="./rtd/_static/logo_mini.png" height="80px" />
</picture>

<br/>

# Modern C++ framework for Symbolic Regression

[![License](https://img.shields.io/github/license/heal-research/operon?style=flat)](https://github.com/heal-research/operon/blob/master/LICENSE)
[![build-linux](https://github.com/heal-research/operon/actions/workflows/build-linux.yml/badge.svg)](https://github.com/heal-research/operon/actions/workflows/build-linux.yml)
[![build-macos](https://github.com/heal-research/operon/actions/workflows/build-macos.yml/badge.svg)](https://github.com/heal-research/operon/actions/workflows/build-macos.yml)
[![build-windows](https://github.com/heal-research/operon/actions/workflows/build-windows.yml/badge.svg)](https://github.com/heal-research/operon/actions/workflows/build-windows.yml)
[![Documentation Status](https://readthedocs.org/projects/operongp/badge/?version=latest)](https://operongp.readthedocs.io/en/latest/?badge=latest)
[![Matrix Channel](https://badges.gitter.im/operongp/gitter.png)](https://gitter.im/operongp/community)

*Operon* is a modern C++ framework for [symbolic regression](https://en.wikipedia.org/wiki/Symbolic_regression) that uses [genetic programming](https://en.wikipedia.org/wiki/Genetic_programming) to explore a hypothesis space of possible mathematical expressions in order to find the best-fitting model for a given [regression target](https://en.wikipedia.org/wiki/Regression_analysis).
Its main purpose is to help develop accurate and interpretable white-box models in the area of [system identification](https://en.wikipedia.org/wiki/System_identification). More in-depth documentation available at https://operongp.readthedocs.io/.

## How does it work?

Broadly speaking, genetic programming (GP) is said to evolve a population of "computer programs" ― [AST](https://en.wikipedia.org/wiki/Abstract_syntax_tree)-like structures encoding behavior for a given problem domain ― following the principles of [natural selection](https://en.wikipedia.org/wiki/Natural_selection). It repeatedly combines random program parts keeping only the best results ― the "fittest". Here, the biological concept of [fitness](https://en.wikipedia.org/wiki/Survival_of_the_fittest) is defined as a measure of a program's ability to solve a certain task.

In symbolic regression, the programs represent mathematical expressions typically encoded as [expression trees](https://en.wikipedia.org/wiki/Binary_expression_tree). Fitness is usually defined as [goodness of fit](https://en.wikipedia.org/wiki/Goodness_of_fit) between the dependent variable and the prediction of a tree-encoded model. Iterative selection of best-scoring models followed by random recombination leads naturally to a self-improving process that is able to uncover patterns in the data:

<p align="center">
    <img src="./rtd/_static/evo.gif"  />
</p>

## Build instructions

The project requires CMake and a C++17 compliant compiler (C++20 if you're on the `cpp20` branch). The recommended way to build Operon is via either [nix](https://github.com/NixOS/nix) or [vcpkg](https://github.com/microsoft/vcpkg).

Check out [https://github.com/heal-research/operon/blob/master/BUILDING.md](BUILD.md) for detailed build instructions and how to enable/disable certain features.

### Nix

First, you have to [install nix](https://nixos.org/download.html) and [enable flakes](https://nixos.wiki/wiki/Flakes).
For a portable install, see [nix-portable](https://github.com/DavHau/nix-portable).

To create a development shell:
```
nix develop github:heal-research/operon --no-write-lock-file
```

To build Operon (a symlink to the nix store called `result` will be created).
```
nix build github:heal-research/operon --no-write-lock-file
```


### Vcpkg

Select the build generator appropriate for your system and point CMake to the `vcpkg.cmake` toolchain file

```
cmake -S . -B build -G "Visual Studio 16 2019" -A x64 \
-DCMAKE_TOOLCHAIN_FILE=..\vcpkg\scripts\buildsystems\vcpkg.cmake \
-DVCPKG_OVERLAY_PORTS=.\ports
```

The file `CMakePresets.json` contains some presets that you may find useful. For using `clang-cl` instead of `cl`, pass `-TClangCL` to the above ([official documentation](https://docs.microsoft.com/en-us/cpp/build/clang-support-cmake?view=msvc-170)).

## Python wrapper

Python bindings for the Operon library are available as a separate project: [PyOperon](https://github.com/heal-research/pyoperon), which also includes a [scikit-learn](https://scikit-learn.org/stable/index.html) compatible regressor.

## Bibtex info

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
@article{DBLP:journals/corr/abs-2107-14351,
    author    = {William G. La Cava and
                 Patryk Orzechowski and
                 Bogdan Burlacu and
                 Fabr{\'{\i}}cio Olivetti de Fran{\c{c}}a and
                 Marco Virgolin and
                 Ying Jin and
                 Michael Kommenda and
                 Jason H. Moore},
    title     = {Contemporary Symbolic Regression Methods and their Relative Performance},
    journal   = {CoRR},
    volume    = {abs/2107.14351},
    year      = {2021},
    url       = {https://arxiv.org/abs/2107.14351},
    eprinttype = {arXiv},
    eprint    = {2107.14351},
    timestamp = {Tue, 03 Aug 2021 14:53:34 +0200},
    biburl    = {https://dblp.org/rec/journals/corr/abs-2107-14351.bib},
    bibsource = {dblp computer science bibliography, https://dblp.org}
}

```
