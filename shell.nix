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
    eigen = pkgs.eigen.overrideAttrs (old: rec {
      src = pkgs.fetchgit {
        url             = "https://gitlab.com/libeigen/eigen.git";
        rev             = "e8f40e4670865b6eda3a4ba7eba2b4cb429e5f9c";
        sha256          = "0gbhmck6ig923qjfwhziphb24j1mqhxlsh9apckljzk9ff92598y";
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
#pkgs.llvmPackages_9.stdenv.mkDerivation {
unstable.gcc9Stdenv.mkDerivation {
    name = "operon-env";
    hardeningDisable = [ "all" ]; 

    buildInputs = with unstable; [
        # python environment for bindings and scripting
        (pkgs.python38.withPackages (ps: with ps; [ pip numpy pandas pybind11 colorama coloredlogs seaborn sphinx recommonmark sphinx_rtd_theme ]))
        # Project dependencies
        bear # generate compilation database
        gdb
        valgrind
        heaptrack
        git
        cmake
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
        graphviz
        cxxopts
    ];
}
