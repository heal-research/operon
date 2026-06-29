{
  stdenv,
  pkgs,
  system,
  enableShared ? true,
  enableTesting ? true,
  enablePappus ? false,
  pappusInclude ? "",
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

  cmakeFlags =
    [
      "--preset ${cmakePreset}"
      "-DUSE_SINGLE_PRECISION=ON"
      "-DBUILD_SHARED_LIBS=${if enableShared then "YES" else "NO"}"
    ]
    ++ pkgs.lib.optionals enablePappus [
      "-DOPERON_ENABLE_PAPPUS=ON"
      "-DPAPPUS_INCLUDE_DIR=${pappusInclude}"
    ];
  cmakeBuildType = "Release";

  nativeBuildInputs = with pkgs; [
    cmake
    git
  ];

  buildInputs =
    (with pkgs; [
      asmjit
      aria-csv
      cpp-sort
      cpptrace
      cxxopts
      eigen
      eve
      fast-float
      fluky
      fmt_11
      glaze
      gtl
      icu
      lbfgs
      libassert
      libdwarf
      mdspan
      microsoft-gsl
      ndsort
      pkg-config
      infix-parser
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
    ++ (with pkgs; pkgs.lib.optionals enablePappus [
      gch-small-vector
    ])
    ++ (
      with pkgs;
      pkgs.lib.optionals enableTesting [
        catch2_3
        nanobench
      ]
    );
}
