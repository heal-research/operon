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
    # ned14 deps
    aria-csv
    byte-lite
    ceres-solver
    cpp-sort
    cpptrace
    cxxopts
    doctest
    eigen
    eve
    fast-float
    fluky
    fmt_11
    gch-small-vector
    icu
    jemalloc
    lbfgs
    libassert
    libdwarf
    mdspan
    microsoft-gsl
    nanobench
    ned14-outcome
    ned14-quickcpplib
    ned14-status-code
    pkg-config
    pratt-parser
    scnlib
    simdutf # required by scnlib
    span-lite
    taskflow
    unordered_dense
    vdt
    vstat
    xxHash
    zstd
  ]);
}
