{ stdenv, pkgs, system, enableShared ? true, enableTesting ? true }:
stdenv.mkDerivation rec {
  name = "operon";
  src = ./.;

  cmakePreset = {
    "x86_64-linux"   = "build-linux";
    "x86_64-darwin"  = "build-linux";
    "aarch64-darwin" = "build-osx";
  }."${system}";

  cmakeFlags = [
    "--preset ${cmakePreset}"
    "-DUSE_SINGLE_PRECISION=ON"
    "-DBUILD_SHARED_LIBS=${if enableShared then "YES" else "NO"}"
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
    eigen
    eve
    fast-float
    fluky
    fmt_11
    icu
    jemalloc
    lbfgs
    libassert
    libdwarf
    mdspan
    microsoft-gsl
    ned14-outcome
    ned14-quickcpplib
    ned14-status-code
    parallel-hashmap
    pkg-config
    pratt-parser
    scnlib
    simdutf # required by scnlib
    singleton
    span-lite
    taskflow
    unordered_dense
    vdt
    vstat
    xxHash
    zstd
  ]) ++ (with pkgs; pkgs.lib.optionals enableTesting [ doctest nanobench ]);
}
