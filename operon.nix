{
  stdenv,
  pkgs,
  system,
  enableShared ? true,
  enableTesting ? true,
  enableAsmjit ? true,
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
    "-DUSE_ASMJIT=${if enableAsmjit then "ON" else "OFF"}"
    "-DBUILD_SHARED_LIBS=${if enableShared then "YES" else "NO"}"
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
      (fmt_12.override { inherit enableShared; })
      glaze
      gtl
      lbfgs
      libassert
      mdspan
      microsoft-gsl
      (if enableShared then ndsort else ndsort-static)
      pappus
      taskflow
      tl-expected
      unordered_dense
      vstat
      (if enableShared then infix-parser else infix-parser-static)
      (xxHash.overrideAttrs (old: {
        cmakeFlags = (old.cmakeFlags or [ ]) ++ pkgs.lib.optionals (!enableShared) [
          "-DBUILD_SHARED_LIBS=OFF"
        ];
      }))
    ])
    ++ (with pkgs; pkgs.lib.optionals enableAsmjit [ (asmjit.override { inherit enableShared; }) ]);

  # Deps only used in .cpp implementation files or by CLI/test binaries; not
  # required at compile time by consumers of operon's headers, and not
  # find_dependency()'d by install-config.cmake.
  buildInputs =
    (with pkgs; [
      cpptrace
      cxxopts
      fast-float
      icu
      libdwarf
      pkg-config
      (scnlib.override { inherit enableShared; })
      simdutf # required by scnlib
      vdt
      zstd
    ])
    # infix-parser's static archive can't resolve its own private link deps
    # (fmt/FastFloat are already pulled in above; lexy isn't otherwise needed).
    ++ (with pkgs; pkgs.lib.optionals (!enableShared) [ foonathan-lexy ])
    ++ (
      with pkgs;
      pkgs.lib.optionals enableTesting [
        catch2_3
        nanobench
      ]
    );
}
