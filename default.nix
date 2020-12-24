let
  pkgs_stable = import <nixos> { };
  pkgs = import <nixos-unstable> { };
  cxxopts = import ./cxxopts.nix; 
  qcachegrind = pkgs.libsForQt5.callPackage ./qcachegrind.nix {};

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
  eigen339 = pkgs.eigen.overrideAttrs (old: rec {
    version = "3.3.9";
    stdenv = pkgs.gcc10Stdenv;
    src = pkgs.fetchFromGitLab {
      owner = "libeigen";
      repo = "eigen";
      rev    = "${version}";
      sha256 = "0m4h9fd5s1pzpncy17r3w0b5a6ywqjajmnr720ndb7fc4bn0dhi4";
    };
    patches = [ ./eigen_include_dir.patch ];
  });
  pybind11_trunk = pkgs.python38Packages.pybind11.overrideAttrs (old: rec {
    stdenv = pkgs.gcc10Stdenv;
    buildInputs = with pkgs; [ eigen339 ];
    version = "2.6.0";
    src = pkgs.fetchFromGitHub {
      repo   = "pybind11";
      owner  = "pybind";
      rev    = "v${version}";
      sha256 = "19rnl4pq2mbh5hmj96cs309wxl51q5yyp364icg26zjm3d0ap834";
    };
  });
  ceres200 = pkgs.ceres-solver.overrideAttrs (old: rec {
    CFLAGS = (old.CFLAGS or "") + "-march=native -O3";
    stdenv = pkgs.gcc10Stdenv;
    buildInputs = with pkgs; [ eigen339 glog ];

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
  eli5 = import ./eli5.nix {
    lib = pkgs.gcc10Stdenv.lib;
    buildPythonPackage = pkgs.python38Packages.buildPythonPackage;
    fetchPypi = pkgs.python38Packages.fetchPypi;
    pythonPackages = pkgs.python38Packages;
  };
  pmlb = import ./pmlb.nix {
    lib = pkgs.gcc10Stdenv.lib;
    buildPythonPackage = pkgs.python38Packages.buildPythonPackage;
    fetchPypi = pkgs.python38Packages.fetchPypi;
    pythonPackages = pkgs.python38Packages;
  };
in
  pkgs.gcc10Stdenv.mkDerivation {
    name = "operon-env";
    hardeningDisable = [ "all" ]; 

    buildInputs = with pkgs; [
        # python environment for bindings and scripting
        pybind11_trunk
        (pkgs.python38.withPackages (ps: with ps; [ pytest pip numpy scipy scikitlearn pandas sympy pyperf colorama coloredlogs seaborn cython jupyterlab ipywidgets grip livereload joblib graphviz dask ]))
        (pkgs_stable.python38.withPackages (ps: with ps; [ sphinx recommonmark sphinx_rtd_theme ]))
        # Project dependencies
        # profiling and debugging
        gdb
        valgrind
        linuxPackages.perf
        #bear
        #tracy
        #bloaty
        #heaptrack
        #hotspot
        cmake
        tbb
        #eigen
        eigen339
        ceres200
        openlibm
        gperftools
        jemalloc
        fmt
        glog
        doctest
#        llvm_10 # code generation
        clang_10
        # visualize profile results
        #pyprof2calltree
        #qcachegrind
        #massif-visualizer
        graphviz
        cxxopts
        #asciinema
        hyperfine
        ninja
        eli5
        pmlb
      ];
    }
