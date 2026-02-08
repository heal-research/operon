{
  stdenv,
  pkgs,
  system,
  enableShared ? true,
  enableTesting ? true,
}:
stdenv.mkDerivation rec {
  name = "operon";
  src = ./.;

  cmakePreset =
    {
      "x86_64-linux" = "build-linux";
      "x86_64-darwin" = "build-linux";
      "aarch64-darwin" = "build-osx";
    }
    ."${system}";

  cmakeFlags = [
    "--preset ${cmakePreset}"
    "-DUSE_SINGLE_PRECISION=ON"
    "-DBUILD_SHARED_LIBS=${if enableShared then "YES" else "NO"}"
  ];
  cmakeBuildType = "Release";

  nativeBuildInputs = with pkgs; [
    cmake
    git
  ];

  buildInputs =
    (with pkgs; [
      aria-csv
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
      pkg-config
      pratt-parser
      (scnlib.overrideAttrs { inherit enableShared; })
      simdutf # required by scnlib
      taskflow
      tl-expected
      unordered_dense
      vdt
      vstat
      xxHash
      zstd
    ])
    ++ (
      with pkgs;
      pkgs.lib.optionals enableTesting [
        doctest
        nanobench
      ]
    );
}
