{
  description = "Operon development environment";

  inputs = {
    flake-utils.url = "github:numtide/flake-utils";
    nixpkgs.url = "github:nixos/nixpkgs/master";
    foolnotion.url = "github:foolnotion/nur-pkg";
    lbfgs.url = "github:foolnotion/lbfgs";
    pratt-parser.url = "github:foolnotion/pratt-parser-calculator";
    vstat.url = "github:heal-research/vstat/cpp20-eve";

    # make everything follow nixpkgs
    foolnotion.inputs.nixpkgs.follows = "nixpkgs";
    lbfgs.inputs.nixpkgs.follows = "nixpkgs";
    pratt-parser.inputs.nixpkgs.follows = "nixpkgs";
    vstat.inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs = { self, flake-utils, nixpkgs, foolnotion, pratt-parser, vstat, lbfgs }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          overlays = [ foolnotion.overlay ];
        };

        stdenv_ = pkgs.llvmPackages_17.stdenv;

        operon = stdenv_.mkDerivation {
          name = "operon";
          src = self;

          cmakeFlags = [
            "-DBUILD_CLI_PROGRAMS=ON"
            "-DBUILD_SHARED_LIBS=${if pkgs.stdenv.hostPlatform.isStatic then "OFF" else "ON"}"
            "-DBUILD_TESTING=OFF"
            "-DCMAKE_BUILD_TYPE=Release"
            "-DUSE_SINGLE_PRECISION=ON"
          ];

          nativeBuildInputs = with pkgs; [ cmake git ];

          buildInputs = (with pkgs; [
            aria-csv
            ceres-solver
            cpp-sort
            cxxopts
            doctest
            eigen
            eve
            fast_float
            fmt
            icu
            jemalloc
            pkg-config
            pratt-parser.packages.${system}.default
            scnlib
            taskflow
            unordered_dense
            vstat.packages.${system}.default
            lbfgs.packages.${system}.default
            ned14-outcome
            ned14-quickcpplib
            ned14-status-code
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
            clang-tools_17
            cppcheck
            include-what-you-use
            cmake-language-server
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
