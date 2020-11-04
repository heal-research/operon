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
      rev = version;
      sha256 = "0r9axsdlmacjlcnax4vkzg86nwf8lsx7wbqdi3wnryaxk0xvdcx6";
    };

    installPhase = old.installPhase + ''
        ${pkgs.cmake}/bin/cmake \
            -DINSTALL_DIR="$out"/lib/cmake/TBB \
            -DSYSTEM_NAME=Linux -DTBB_VERSION_FILE="$out"/include/tbb/tbb_stddef.h \
            -P cmake/tbb_config_installer.cmake
    '';
  });
  eigen_trunk = pkgs.eigen.overrideAttrs (old: rec {
    version = "3.3.90";
    stdenv = pkgs.gcc10Stdenv;
    src = pkgs.fetchFromGitLab {
      owner = "libeigen";
      repo = "eigen";
      rev    = "28aef8e816faadc0e51afbfe3fa91f10f477535d";
      sha256 = "151bkpn7pkmjglfn4kbdh442g94rjv33n13vy1fgzs9mpjlhmxj9";
    };
    patches = [ ./eigen_include_dir.patch ];
  });
  pybind11_trunk = pkgs.python38Packages.pybind11.overrideAttrs (old: rec {
    stdenv = pkgs.gcc10Stdenv;
    buildInputs = with pkgs; [ eigen_trunk ];
    version = "2.6.0";
    src = pkgs.fetchFromGitHub {
      repo   = "pybind11";
      owner  = "pybind";
      rev    = "v${version}";
      sha256 = "19rnl4pq2mbh5hmj96cs309wxl51q5yyp364icg26zjm3d0ap834";
    };
  });
  ceres_trunk = pkgs.ceres-solver.overrideAttrs (old: rec {
    CFLAGS = (old.CFLAGS or "") + "-march=native -O3";
    stdenv = pkgs.gcc10Stdenv;
    buildInputs = with pkgs; [ eigen_trunk glog ];

    version = "2.0.0";
    src = pkgs.fetchFromGitHub {
      repo   = "ceres-solver";
      owner  = "ceres-solver";
      rev    = "${version}";
      sha256 = "05wpvz461kpbx221s1idsg5lx4hzz28lnnvrniygclrmr0d9s7xg";
    };
    enableParallelBuilding = true;
    cmakeFlags = [ "-DCMAKE_BUILD_TYPE=Release" "-DCXX11=ON" "-DTBB=ON" "-DOPENMP=OFF" "-DBUILD_SHARED_LIBS=ON -DBUILD_EXAMPLES=FALSE -DBUILD_TESTING=FALSE" ];
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
        (pkgs.python38.withPackages (ps: with ps; [ pytest pip numpy scipy scikitlearn pandas sympy pyperf colorama coloredlogs seaborn cython jupyterlab ipywidgets grip livereload joblib graphviz ]))
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
        eigen_trunk
        ceres_trunk
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
