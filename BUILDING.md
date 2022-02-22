# Building with CMake

## Build

This project doesn't require any special command-line flags to build to keep
things simple.

### Using nix
The recommended way to build operon is to use nix package manager.
```sh
nix develop --extra-experimental-features nix-command --extra-experimental-features flakes
```
It downloads the required packages in the correct version (as specified in flake.nix) and starts a 'nix develop' shell.

Here are the steps for building in release mode with a single-configuration
generator, like the Unix Makefiles one:

```sh
cmake -S . -B build -D CMAKE_BUILD_TYPE=Release
cmake --build build
```

Here are the steps for building in release mode with a multi-configuration
generator, like the Visual Studio ones:

```sh
cmake -S . -B build
cmake --build build --config Release
```

### Using vcpkg
Alternatively you can use vcpkg.
You require cmake version >= 3.20 because we are using vcpkg ports.
You can install a recent cmake version via vckpg:
```sh
./vcpkg install vcpkg-cmake
```

After installation of this cmake version you can use it with the following options:
```sh
.<vcpkg-dir>/downloads/tools/cmake-3.21.1-linux/cmake-3.21.1-linux-x86_64/bin/cmake \
  --preset build-ubuntu-vcpkg -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE=../vcpkg/scripts/buildsystems/vcpkg.cmake
```

Additional build flags such as ```-DBUILD_CLI_PROGRAMS=ON -DUSE_SINGLE_PRECISION=ON -DUSE_OPENLIBM=ON``` may be added to the command.

## Install

This project doesn't require any special command-line flags to install to keep
things simple. As a prerequisite, the project has to be built with the above
commands already.

The below commands require at least CMake 3.15 to run, because that is the
version in which [Install a Project][1] was added.

Here is the command for installing the release mode artifacts with a
single-configuration generator, like the Unix Makefiles one:

```sh
cmake --install build
```

Here is the command for installing the release mode artifacts with a
multi-configuration generator, like the Visual Studio ones:

```sh
cmake --install build --config Release
```

[1]: https://cmake.org/cmake/help/latest/manual/cmake.1.html#install-a-project
