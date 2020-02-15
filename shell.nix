let
    pkgs = import <nixpkgs> {};
    cxxopts = import ./cxxopts.nix; 
    qcachegrind = pkgs.libsForQt5.callPackage ./qcachegrind.nix {};
    tbb = pkgs.tbb.overrideAttrs (old: rec {
        installPhase = old.installPhase + ''
        ${pkgs.cmake}/bin/cmake \
            -DINSTALL_DIR="$out"/lib/cmake/TBB \
            -DSYSTEM_NAME=Linux -DTBB_VERSION_FILE="$out"/include/tbb/tbb_stddef.h \
            -P cmake/tbb_config_installer.cmake
    '';
    });
    eigen = pkgs.eigen.overrideAttrs (old: rec {
        src = pkgs.fetchgit {
          url             = "https://gitlab.com/libeigen/eigen.git";
          rev             = "776960024585b907acc4abc3c59aef605941bb75";
          sha256          = "119h38g9j02jhgf68xm8zzvbswi95amq4nqjqi3nvi6ds6b0k2yk";
          fetchSubmodules = false;
        };
        patches = [ ./include-dir.patch ];
    });
    ceres-solver = pkgs.ceres-solver.overrideAttrs (old: rec {
        cmakeFlags = [ "-DCMAKE_BUILD_TYPE=Release" "-DCXX11=ON" "-DTBB=ON" "-DOPENMP=OFF" "-DBUILD_SHARED_LIBS=ON" ];
    });
    fmt = pkgs.fmt.overrideAttrs(old: { outputs = [ "out" ]; });
    gcc = { 
      arch = "znver2"; 
      tune = "znver2"; 
    };
in
pkgs.gcc9Stdenv.mkDerivation {
    name = "operon-env";
    hardeningDisable = [ "all" ]; 

    buildInputs = with pkgs; [
        # python environment for bindings and scripting
        (pkgs.python37.withPackages (ps: with ps; [ pip numpy pandas cython scikitlearn pybind11 colorama coloredlogs seaborn ]))
        # Project dependencies
        bear # generate compilation database
        gdb
        valgrind
        git
        cmake
        cxxopts
        eigen
        openlibm
        gperftools
        jemalloc
        fmt
        glog
        ceres-solver
        tbb
        catch2
        # visualize profile results
        qcachegrind
        massif-visualizer
    ];
}
