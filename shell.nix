let
  pkgs = import <nixpkgs> {};
  unstable = import <nixos-unstable> {};
  cxxopts = import ./cxxopts.nix; 
  qcachegrind = unstable.libsForQt5.callPackage ./qcachegrind.nix {};
  tbb = unstable.tbb.overrideAttrs (old: rec {
    installPhase = old.installPhase + ''
        ${pkgs.cmake}/bin/cmake \
            -DINSTALL_DIR="$out"/lib/cmake/TBB \
            -DSYSTEM_NAME=Linux -DTBB_VERSION_FILE="$out"/include/tbb/tbb_stddef.h \
            -P cmake/tbb_config_installer.cmake
    '';
  });
#  eigen_trunk = unstable.eigen.overrideAttrs (old: rec {
#    src = unstable.fetchgit {
#      url             = "https://gitlab.com/libeigen/eigen.git";
#      rev             = "2fd8a5a08fece826d211a6f34d777bb65f6b4562";
#      sha256          = "045ir8vc41cd8qf6www9pblz6hl41zfbbx4mi21b55y1kr5fcxla";
#      fetchSubmodules = false;
#    };
#    patches = [ ./include-dir.patch ];
#  });
  ceres_trunk = unstable.ceres-solver.overrideAttrs (old: rec {
    #buildInputs = [ eigen_trunk unstable.glog ];
    src = unstable.fetchgit {
      url             = "https://github.com/ceres-solver/ceres-solver.git";
      rev             = "323cc55bb92a513924e566f487b54556052a716f";
      sha256          = "18m6kr3mzhgb72zvxvzv2d3hbl6zzhgviqgikwhx3vdfk9cx7qlx";
      fetchSubmodules = false;
    };
    cmakeFlags = [ "-DCMAKE_BUILD_TYPE=Release" "-DCXX11=ON" "-DTBB=ON" "-DOPENMP=OFF" "-DBUILD_SHARED_LIBS=ON" ];
  });
  fmt = unstable.fmt.overrideAttrs(old: { outputs = [ "out" ]; });
in
#unstable.llvmPackages_10.stdenv.mkDerivation {
unstable.gcc10Stdenv.mkDerivation {
    name = "operon-env";
    hardeningDisable = [ "all" ]; 

    buildInputs = with unstable; [
        # python environment for bindings and scripting
        (pkgs.python38.withPackages (ps: with ps; [ pip numpy pandas pybind11 colorama coloredlogs seaborn sphinx recommonmark sphinx_rtd_theme ]))
        # Project dependencies
        bear # generate compilation database
        gdb
        valgrind
        linuxPackages.perf
        bloaty
        #heaptrack
        #hotspot
        git
        cmake
        tbb
        eigen
        #eigen_trunk
        ceres_trunk
        openlibm
        gperftools
        jemalloc
        fmt
        glog
        catch2
        llvm_10 # code generation
        clang_10
        # visualize profile results
        #qcachegrind
        #massif-visualizer
        graphviz
        cxxopts
      ];
    }
