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
            aria-csv
            cpp-sort
            cxxopts
            doctest
            eigen
            eve
            fast_float
            fmt_8
            git
            openlibm
            pkg-config
            pratt-parser.defaultPackage.${system}
            scnlib
            taskflow
            unordered_dense
            vstat.packages.${system}.default
            xxHash
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
            #hotspot
            gdb
            graphviz
            hyperfine
            linuxPackages.perf
            pyprof2calltree
            qcachegrind
            seer
            valgrind
          ]);

          shellHook = ''
            alias bb="cmake --build build -j"
          '';
        };
      });
}
