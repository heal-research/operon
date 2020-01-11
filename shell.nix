let
    pkgs = import <nixpkgs> {};
    cxxopts = import ./cxxopts.nix; 
    tbb = pkgs.tbb.overrideAttrs (old: rec {
        installPhase = old.installPhase + ''
        ${pkgs.cmake}/bin/cmake \
            -DINSTALL_DIR="$out"/lib/cmake/TBB \
            -DSYSTEM_NAME=Linux -DTBB_VERSION_FILE="$out"/include/tbb/tbb_stddef.h \
            -P cmake/tbb_config_installer.cmake
    '';
    });
    ceres-solver = pkgs.ceres-solver.overrideAttrs (old: rec {
        cmakeFlags = [ "-DCMAKE_BUILD_TYPE=Release" "-DCXX11=ON" "-DTBB=ON" "-DBUILD_SHARED_LIBS=ON" ];
    });
    fmt = pkgs.fmt.overrideAttrs(old: { outputs = [ "out" ]; });
    gcc = { 
      arch = "znver2"; 
      tune = "znver2"; 
    };
#    _python37 = (let
#      packageOverrides = self: super: {
#        numexpr = super.numexpr.overridePythonAttrs({
#          src = super.fetchPypi { 
#            pname = "numexpr";
#            version = "2.7.1";
#            sha256 = "1c82z0zx0542j9df6ckjz6pn1g13b21hbza4hghcw6vyhbckklmh";
#          };
#        });
#        numpy = super.numpy.overridePythonAttrs({
#          src = super.fetchPypi { 
#            pname = "numpy";
#            version = "1.17.5";
#            extension = "zip";
#            sha256 = "0hn38q24fnkd7yw6k8yrshji0cshypk5lwmid8yawqkzc6k7nl0n";
#          };
#        });
#      };
#    in pkgs.python37.override { inherit packageOverrides; }).withPackages (ps: with ps; [ pip numpy pandas cython scikitlearn pybind11 colorama coloredlogs ]);
in
pkgs.gcc9Stdenv.mkDerivation {
    name = "operon-env";
    hardeningDisable = [ "all" ]; 

    buildInputs = with pkgs; [
        # python environment for bindings and scripting
        (pkgs.python37.withPackages (ps: with ps; [ pip numpy pandas cython scikitlearn pybind11 colorama coloredlogs ]))
        # Project dependencies
        bear # generate compilation database
        git
        cmake
        cxxopts
        eigen
        fmt
        glog
        ceres-solver
        tbb
        catch2
    ];
}
