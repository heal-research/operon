{ stdenv, pkgs }:
stdenv.mkDerivation {
  name = "operon";
  src = ./.;

  enableShared = true;

  cmakeFlags = [
    "--preset ${if pkgs.stdenv.hostPlatform.isx86_64 then "build-linux" else "build-osx"}"
    "-DUSE_SINGLE_PRECISION=ON"
  ];
  cmakeBuildType = "Release";

  nativeBuildInputs = with pkgs; [ cmake git ];

  buildInputs = (with pkgs; [
    aria-csv
    armadillo
    blaze
    ceres-solver
    cpp-sort
    cxxopts
    doctest
    eigen
    eve
    fast_float
    fastor
    fmt
    icu
    jemalloc
    cpptrace
    libassert
    libdwarf
    mdspan
    pkg-config
    pratt-parser
    simdutf_4 # required by scnlib
    scnlib
    sleef
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
    xad
    xsimd
    xxHash
    zstd
  ]);
}
