# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

let
  pkgs = import ~/nix/nixpkgs {
      #localSystem = {
      #  gcc.arch = "znver2";
      #  gcc.tune = "znver2";
      #  system = "x86_64-linux";
      #};
  };

  eigen = pkgs.eigen.overrideAttrs (old: rec {
    version = "3.4";
    stdenv = pkgs.gcc11Stdenv;
    src = builtins.fetchGit {
      url = "https://gitlab.com/libeigen/eigen";
      ref = "3.4";
      rev = "972cf0c28a8d2ee0808c1277dea2c5c206591ce6";
    };
    patches = [];
    cmakeFlags = [ "-DCMAKE_PREFIX_PATH=$out" "-DINCLUDE_INSTALL_DIR=include/eigen3" ];
  });

  ceres-solver = pkgs.ceres-solver.overrideAttrs (old: rec {
    CFLAGS = (old.CFLAGS or "") + "-march=native -O3";
    stdenv = pkgs.gcc11Stdenv;
    buildInputs = with pkgs; [ eigen glog ];

    version = "2.0.0";
    src = pkgs.fetchFromGitHub {
      repo   = "ceres-solver";
      owner  = "ceres-solver";
      rev    = "ec4f2995bbde911d6861fb5c9bb7353ad796e02b";
      sha256 = "sha256-4sWOipg90G9dufmRzSSwwmvFpkgp3KDuVsBANC5YvfU=";
    };
    enableParallelBuilding = true;
    cmakeFlags = [ "-DCMAKE_BUILD_TYPE=Release" "-DCXX11=ON" "-DTBB=OFF" "-DOPENMP=OFF" "-DBUILD_SHARED_LIBS=OFF" "-DBUILD_EXAMPLES=FALSE" "-DBUILD_TESTING=FALSE" ];
  });

  fmt = pkgs.fmt.overrideAttrs(old: rec { 
    outputs = [ "out" ];

    cmakeFlags = [ "-DBUILD_SHARED_LIBS=ON" "-DFMT_TEST=OFF" "-DFMT_CUDA_TEST=OFF" "-DFMT_FUZZ=OFF" ];
  });
in
  pkgs.gcc11Stdenv.mkDerivation {
    name = "operon-env";
    hardeningDisable = [ "all" ]; 

    impureUseNativeOptimizations = true;

    nativeBuildInputs = with pkgs; [ bear cmake clang_12 clang-tools ];

    buildInputs = with pkgs; [
        # python environment for bindings and scripting
        (python39.withPackages (ps: with ps; [ pybind11 pytest pip numpy scipy scikitlearn pandas sympy pyperf colorama coloredlogs seaborn cython jupyterlab ipywidgets grip livereload joblib graphviz dask sphinx recommonmark sphinx_rtd_theme ]))
        # Project dependencies and utils for profiling and debugging
        ceres-solver
        cmake
        cxxopts
        diff-so-fancy
        doctest
        eigen
        eli5
        fmt
        gdb
        glog
        gperftools
        graphviz
        hyperfine
        jemalloc
        linuxPackages.perf
        mimalloc
        ninja
        openlibm
        pmlb
        taskflow
        valgrind
      ];
    }
