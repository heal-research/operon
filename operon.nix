{
  stdenv,
  pkgs,
  system,
  enableShared ? true,
  enableTesting ? true,
  enablePappus ? false,
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
    ];
  cmakeBuildType = "Release";

  nativeBuildInputs = with pkgs; [
    cmake
    git
  ];

  # Deps that cmake/install-config.cmake `find_dependency()`s for consumers of
  # `find_package(operon CONFIG REQUIRED)` (e.g. pyoperon) must be resolvable
  # via CMAKE_PREFIX_PATH downstream too, regardless of whether they show up
  # in operon's public C++ headers — install-config.cmake calls these
  # unconditionally, even the ones it labels "needed for static library".
  propagatedBuildInputs =
    (with pkgs; [
      aria-csv
      cpp-sort
      eigen
      eve
      fluky
      fmt_11
      gtl
      lbfgs
      libassert
      mdspan
      microsoft-gsl
      taskflow
      tl-expected
      unordered_dense
      vstat
      infix-parser
    ])
    ++ (with pkgs; pkgs.lib.optionals enablePappus [ pappus ]);

  # Deps only used in .cpp implementation files or by CLI/test binaries; not
  # required at compile time by consumers of operon's headers, and not
  # find_dependency()'d by install-config.cmake.
  buildInputs =
    (with pkgs; [
      asmjit
      cpptrace
      cxxopts
      fast-float
      glaze
      icu
      libdwarf
      ndsort
      pkg-config
      (scnlib.overrideAttrs { inherit enableShared; })
      simdutf # required by scnlib
      vdt
      xxHash
      zstd
    ])
    ++ (
      with pkgs;
      pkgs.lib.optionals enableTesting [
        catch2_3
        nanobench
      ]
    );
}
