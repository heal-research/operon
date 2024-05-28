{
  description = "Operon development environment";

  inputs = {
    flake-utils.url = "github:numtide/flake-utils";
    nixpkgs.url = "github:nixos/nixpkgs/master";
    foolnotion.url = "github:foolnotion/nur-pkg";
    lbfgs.url = "github:foolnotion/lbfgs";
    pratt-parser.url = "github:foolnotion/pratt-parser-calculator";
    vstat.url = "github:heal-research/vstat";
    vdt.url = "github:foolnotion/vdt/master";

    # make everything follow nixpkgs
    foolnotion.inputs.nixpkgs.follows = "nixpkgs";
    lbfgs.inputs.nixpkgs.follows = "nixpkgs";
    pratt-parser.inputs.nixpkgs.follows = "nixpkgs";
    vstat.inputs.nixpkgs.follows = "nixpkgs";
    vdt.inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs = { self, flake-utils, nixpkgs, foolnotion, pratt-parser, vdt, vstat, lbfgs }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          overlays = [ foolnotion.overlay ];
        };

        stdenv_ = pkgs.llvmPackages_18.stdenv;

        operon = stdenv_.mkDerivation {
          name = "operon";
          src = self;

          enableShared = true;

          cmakeFlags = [
            "--preset ${if pkgs.stdenv.hostPlatform.isx86_64 then "build-linux" else "build-osx"}"
            "-DCMAKE_BUILD_TYPE=Release"
            "-DUSE_SINGLE_PRECISION=ON"
          ];

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
            pratt-parser.packages.${system}.default
            simdutf_4 # required by scnlib
            scnlib
            sleef
            taskflow
            unordered_dense
            vdt.packages.${system}.default
            vstat.packages.${system}.default
            lbfgs.packages.${system}.default
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
        };

      in rec {
        packages = {
          default = operon.overrideAttrs(old: {
            cmakeFlags = old.cmakeFlags ++ [
              "-DBUILD_CLI_PROGRAMS=ON"
            ];
          });

          library = operon.overrideAttrs(old: {
            enableShared = true;
          });

          library-static = operon.overrideAttrs(old: {
            enableShared = false;
          });
        };

        apps.operon-gp = {
          type = "app";
          program = "${packages.default}/bin/operon_gp";
        };

        apps.operon-nsgp = {
          type = "app";
          program = "${packages.default}/bin/operon_nsgp";
        };

        apps.parse-model = {
          type = "app";
          program = "${packages.default}/bin/operon_parse_model";
        };

        devShells.default = stdenv_.mkDerivation {
          name = "operon";

          nativeBuildInputs = operon.nativeBuildInputs ++ (with pkgs; [
            clang-tools_18
            cppcheck
            include-what-you-use
            #cmake-language-server
          ]);

          buildInputs = operon.buildInputs ++ (with pkgs; [
            gdb
            gcc13
            graphviz
            hyperfine
            linuxPackages_latest.perf
            seer
            valgrind
            hotspot
          ]);

          shellHook = ''
            export LD_LIBRARY_PATH=$CMAKE_LIBRARY_PATH
            alias bb="cmake --build build -j"
          '';
        };
      });
}
