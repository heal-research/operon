let
  pkgs = import <nixos-unstable> {};
  cxxopts = import ./cxxopts.nix; 
  tracy = import ./tracy.nix;
  qcachegrind = pkgs.libsForQt5.callPackage ./qcachegrind.nix {};
  tbb = pkgs.tbb.overrideAttrs (old: rec {
    installPhase = old.installPhase + ''
        ${pkgs.cmake}/bin/cmake \
            -DINSTALL_DIR="$out"/lib/cmake/TBB \
            -DSYSTEM_NAME=Linux -DTBB_VERSION_FILE="$out"/include/tbb/tbb_stddef.h \
            -P cmake/tbb_config_installer.cmake
    '';
  });
  eigen_trunk = pkgs.eigen.overrideAttrs (old: rec {
    src = pkgs.fetchgit {
      url             = "https://gitlab.com/libeigen/eigen.git";
      rev             = "f566724023e1a82be7fecfe0639e908772d3cea6";
      sha256          = "055x45z5nh14kh7vig8kl23mp3zmsm3c6924hnlghia6qpmm4vc1";
      fetchSubmodules = false;
    };
    #patches = [ ./eigen_include_dir.patch ];
    #patches = [ ./eigen_include_dir_old.patch ];
    patches = [ ./eigen_include_dir_oldest.patch ];
  });
  ceres_trunk = pkgs.ceres-solver.overrideAttrs (old: rec {
    buildInputs = [ eigen_trunk pkgs.glog ];
    src = pkgs.fetchgit {
      url             = "https://github.com/ceres-solver/ceres-solver.git";
      rev             = "e39d9ed1d60dfeb58dd2a0df4622c683f87b28e3";
      sha256          = "0f9h87hhbnk7j15sf97dw9hnyf3d58r0b6hy1sldrmpgl827x9w0";
      fetchSubmodules = false;
    };
    cmakeFlags = [ "-DCMAKE_BUILD_TYPE=Release" "-DCXX11=ON" "-DTBB=ON" "-DOPENMP=OFF" "-DBUILD_SHARED_LIBS=ON -DBUILD_EXAMPLES=FALSE" ];
  });
  fmt = pkgs.fmt.overrideAttrs(old: { outputs = [ "out" ]; });
in
#unstable.llvmPackages_10.stdenv.mkDerivation {
pkgs.gcc10Stdenv.mkDerivation {
    name = "operon-env";
    hardeningDisable = [ "all" ]; 

    buildInputs = with pkgs; [
        # python environment for bindings and scripting
        (pkgs.python38.withPackages (ps: with ps; [ pip numpy pandas pybind11 pyperf colorama coloredlogs seaborn sphinx recommonmark sphinx_rtd_theme ]))
        # Project dependencies
        ccls # completion vim
        bear # generate compilation database
        # profiling and debugging
        gdb
        valgrind
        linuxPackages.perf
        tracy
        #bloaty
        #heaptrack
        #hotspot
        git
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
        llvm_10 # code generation
        clang_10
        # visualize profile results
        qcachegrind
        #massif-visualizer
        graphviz
        cxxopts
      ];
    }
