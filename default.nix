# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

let
  pkgs = import ~/nix/nixpkgs {
  };
in
  pkgs.gcc11Stdenv.mkDerivation {
    name = "operon-env";
    hardeningDisable = [ "all" ];

    impureUseNativeOptimizations = true;

    nativeBuildInputs = with pkgs; [ cmake clang_12 clang-tools ];

    buildInputs = with pkgs; [
        # python environment for bindings and scripting
        (python39.withPackages (ps: with ps; [ pybind11 pytest pip numpy scipy scikitlearn pandas sympy pyperf colorama coloredlogs seaborn cython jupyterlab ipywidgets grip livereload joblib graphviz sphinx recommonmark sphinx_rtd_theme ]))
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
