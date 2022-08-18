{
  description = "Operon development environment";

  inputs = {
    flake-utils.url = "github:numtide/flake-utils";
    nixpkgs.url = "github:nixos/nixpkgs/master";
    pratt-parser.url = "github:foolnotion/pratt-parser-calculator";
    vstat.url = "github:heal-research/vstat/cpp20-eve";
    foolnotion.url = "github:foolnotion/nur-pkg";
  };

  outputs = { self, flake-utils, nixpkgs, foolnotion, pratt-parser, vstat }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          overlays = [ foolnotion.overlay ];
        };

        operon = pkgs.stdenv.mkDerivation {
          name = "operon";
          src = self;

          cmakeFlags = [
            "-DBUILD_CLI_PROGRAMS=ON"
            "-DBUILD_SHARED_LIBS=${if pkgs.stdenv.hostPlatform.isStatic then "OFF" else "ON"}"
            "-DBUILD_TESTING=OFF"
            "-DCMAKE_BUILD_TYPE=Release"
            "-DUSE_OPENLIBM=ON"
            "-DUSE_SINGLE_PRECISION=ON"
          ];

          nativeBuildInputs = with pkgs; [ cmake ];

          buildInputs = (with pkgs; [
            cxxopts
            doctest
            eigen
            fmt_8
            git
            openlibm
            pkg-config
            xxHash
            taskflow
            pratt-parser.defaultPackage.${system}
            vstat.defaultPackage.${system}
            # foolnotion overlay
            aria-csv
            cpp-sort
            eve
            fast_float
            robin-hood-hashing
            scnlib
          ]);
        };

      in rec {
        packages = {
          default = operon.overrideAttrs(old: {
            cmakeFlags = old.cmakeFlags ++ [
              "-DCMAKE_CXX_FLAGS=${
                if pkgs.stdenv.hostPlatform.isx86_64 then "-march=x86-64-v3" else ""
              }"
            ];
          });

          operon-generic = operon.overrideAttrs(old: {
            cmakeFlags = old.cmakeFlags ++ [
              "-DCMAKE_CXX_FLAGS=${
                if pkgs.stdenv.hostPlatform.isx86_64 then "-march=x86-64" else ""
              }"
            ];
          });
        };

        devShells.default = pkgs.mkShell {
          nativeBuildInputs = operon.nativeBuildInputs ++ (with pkgs; [
            bear
            clang_14
            clang-tools
            cppcheck
            include-what-you-use
          ]);

          buildInputs = operon.buildInputs ++ (with pkgs; [
            gdb
            hotspot
            hyperfine
            valgrind
            linuxPackages.perf
            graphviz
            seer
          ]);

          shellHook = ''
            alias bb="cmake --build build -j"
          '';
        };
      });
}
