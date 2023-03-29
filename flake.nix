{
  description = "Operon development environment";

  inputs = {
    flake-utils.url = "github:numtide/flake-utils";
    nixpkgs.url = "github:nixos/nixpkgs/staging-next";
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

        stdenv_ = pkgs.overrideCC pkgs.llvmPackages_15.stdenv (
          pkgs.clang_15.override { gccForLibs = pkgs.gcc12.cc; }
        );

        operon = stdenv_.mkDerivation {
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

          nativeBuildInputs = with pkgs; [ cmake gcc12 ];

          buildInputs = (with pkgs; [
            aria-csv
            ceres-solver
            cpp-sort
            cxxopts
            doctest
            eigen
            eve
            fast_float
            fmt_9
            jemalloc
            ninja
            openlibm
            pkg-config
            pratt-parser.packages.${system}.default
            scnlib
            taskflow
            unordered_dense
            vstat.packages.${system}.default
            xxhash_cpp
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
            clang-tools
            cppcheck
            include-what-you-use
          ]);

          buildInputs = operon.buildInputs ++ (with pkgs; [
            gdb
            graphviz
            hotspot
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
