let
  pkgs_stable = import <nixos> { };
  pkgs = import <nixos-unstable> { };
  cxxopts = import ./cxxopts.nix; 
  tracy = import ./tracy.nix;
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
      rev             = "2279f2c62f1e5c0dfaa500e637c92d528acd85b1";
      sha256          = "1rdl4vkfn9r9z9x7arb8y8sf477znp6xjyzwfl9p5hs8a3a717gm";
    };
    patches = [ ./eigen_include_dir.patch ];
  });
  pybind11_trunk = pkgs.python38Packages.pybind11.overrideAttrs (old: rec {
    stdenv = pkgs.gcc10Stdenv;
    buildInputs = with pkgs; [ eigen_trunk ];
    src = pkgs.fetchFromGitHub {
      repo   = "pybind11";
      owner  = "pybind";
      rev    = "b6f37f67ac6c6d2ec2fa35ca3816b9de7da9fcbd";
      sha256 = "1v3nliadmgfs8pkm5mrf6z0071588xcwdd2hqnlh8sfvhf2j24zf";
    };
    patches = [ ./pybind11_include.patch ];
  });
  ceres_trunk = pkgs.ceres-solver.overrideAttrs (old: rec {
    CFLAGS = (old.CFLAGS or "") + "-march=native -O3";
    stdenv = pkgs.gcc10Stdenv;
    buildInputs = with pkgs; [ eigen_trunk glog ];

    version = "2.0.0";
    src = pkgs.fetchFromGitHub {
      repo   = "ceres-solver";
      owner  = "ceres-solver";
      rev    = "65c397daeca77da53d16e73720b9a17edd6757ab";
      sha256 = "1v5ahl0ni5kz58ppmaarznbjwlhg7w7y06nyakc2jzjxqbzwv4vx";
    };
    enableParallelBuilding = true;
    cmakeFlags = [ "-DCMAKE_BUILD_TYPE=Release" "-DCXX11=ON" "-DTBB=ON" "-DOPENMP=OFF" "-DBUILD_SHARED_LIBS=ON -DBUILD_EXAMPLES=FALSE -DBUILD_TESTING=FALSE" ];
  });
  fmt = pkgs.fmt.overrideAttrs(old: rec { 
    #version = "7.0.3";
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
        pybind11_trunk
        (pkgs.python38.withPackages (ps: with ps; [ pip numpy scipy scikitlearn pandas sympy pyperf colorama coloredlogs seaborn cython jupyterlab ipywidgets grip livereload ]))
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
      ];
    }
