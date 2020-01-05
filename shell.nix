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
in
pkgs.gcc9Stdenv.mkDerivation {
    name = "operon-env";
    hardeningDisable = [ "all" ]; 

    buildInputs = with pkgs; [
        # Project dependencies
        bear # generate compilation database
        git
        python38
        python38Packages.pybind11
        cmake
        cxxopts
        eigen
        fmt
        ceres-solver
        tbb
        catch2
    ];
}
