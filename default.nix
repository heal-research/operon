let
  pkgs = import <nixos-unstable> { };

  tbb = pkgs.tbb.overrideAttrs (old: rec {
    version = "2020_U3";

    src = pkgs.fetchFromGitHub {
      owner = "01org";
      repo = "tbb";
      rev = "${version}";
      sha256 = "0r9axsdlmacjlcnax4vkzg86nwf8lsx7wbqdi3wnryaxk0xvdcx6";
    };

    installPhase = old.installPhase + ''
        ${pkgs.cmake}/bin/cmake \
            -DINSTALL_DIR="$out"/lib/cmake/TBB \
            -DSYSTEM_NAME=Linux -DTBB_VERSION_FILE="$out"/include/tbb/tbb_stddef.h \
            -P cmake/tbb_config_installer.cmake
    '';
  });
  eigen = pkgs.eigen.overrideAttrs (old: rec {
    version = "3.4";
    stdenv = pkgs.gcc10Stdenv;
    src = builtins.fetchGit {
      url = "https://gitlab.com/libeigen/eigen";
      ref = "3.4";
      rev = "92b2167e3a2d54174fa8c453d774b273f3f75cb7";
    };
    patches = [];
    cmakeFlags = [ "-DCMAKE_PREFIX_PATH=$out" "-DINCLUDE_INSTALL_DIR=include/eigen3" ];
  });
  ceres-solver = pkgs.ceres-solver.overrideAttrs (old: rec {
    CFLAGS = (old.CFLAGS or "") + "-march=native -O3";
    stdenv = pkgs.gcc10Stdenv;
    buildInputs = with pkgs; [ eigen glog ];

    version = "2.0.0";
    src = pkgs.fetchFromGitHub {
      repo   = "ceres-solver";
      owner  = "ceres-solver";
      rev    = "e84cf10e13633618a780543e83c117a84316b790";
      sha256 = "04w1gip6ag6fjs89kds5sgpr6djnfsfwjyhhdcx7mrrdz8lva077";
    };
    enableParallelBuilding = true;
    cmakeFlags = [ "-DCMAKE_BUILD_TYPE=Release" "-DCXX11=ON" "-DTBB=OFF" "-DOPENMP=OFF" "-DBUILD_SHARED_LIBS=ON -DBUILD_EXAMPLES=FALSE -DBUILD_TESTING=FALSE" ];
  });
  fmt = pkgs.fmt.overrideAttrs(old: rec { 
    outputs = [ "out" ];

    cmakeFlags = [
      "-DBUILD_SHARED_LIBS=ON"
      "-DFMT_TEST=OFF"
      "-DFMT_CUDA_TEST=OFF"
      "-DFMT_FUZZ=OFF"
    ];
  });
in
  pkgs.gcc10Stdenv.mkDerivation {
    name = "operon-env";
    hardeningDisable = [ "all" ]; 

    buildInputs = with pkgs; [
        # python environment for bindings and scripting
        (pkgs.python38.withPackages (ps: with ps; [ pybind11 pytest pip numpy scipy scikitlearn pandas sympy pyperf colorama coloredlogs seaborn cython jupyterlab ipywidgets grip livereload joblib graphviz dask sphinx recommonmark sphinx_rtd_theme ]))
        # Project dependencies and utils for profiling and debugging
        gdb
        valgrind
        linuxPackages.perf
        cmake
        tbb
        eigen
        ceres-solver
        openlibm
        gperftools
        jemalloc
        mimalloc
        fmt
        glog
        doctest
        clang_10
        graphviz
        cxxopts
        ninja
        eli5
        pmlb
      ];
    }
