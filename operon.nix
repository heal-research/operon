{ stdenv, pkgs, system }:
stdenv.mkDerivation rec {
  name = "operon";
  src = ./.;

  enableShared = true;

  cmakePreset = {
    "x86_64-linux"   = "build-linux";
    "x86_64-darwin"  = "build-linux";
    "aarch64-darwin" = "build-osx";
  }."${system}"; 

  cmakeFlags = [
    "--preset ${cmakePreset}"
    "-DUSE_SINGLE_PRECISION=ON"
  ];
  cmakeBuildType = "Release";

  nativeBuildInputs = with pkgs; [ cmake git ];

  buildInputs = (with pkgs; [
    aria-csv
    ceres-solver
    cpp-sort
    cxxopts
    doctest
    eigen
    eve
    fast-float
    fmt
    icu
    jemalloc
    cpptrace
    libassert
    libdwarf
    mdspan
    pkg-config
    pratt-parser
    simdutf # required by scnlib
    scnlib
    taskflow
    unordered_dense
    vdt
    vstat
    lbfgs
    # ned14 deps
    byte-lite
    span-lite
    ned14-outcome
    ned14-quickcpplib
    ned14-status-code
    xxHash
    zstd
  ]);
}
