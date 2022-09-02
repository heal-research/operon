{
  description = "Operon development environment";

  inputs.flake-utils.url = "github:numtide/flake-utils";
  inputs.foolnotion.url = "github:foolnotion/nur-pkg";
  inputs.nixpkgs.url = "github:nixos/nixpkgs/master";
  inputs.pratt-parser.url = "github:foolnotion/pratt-parser-calculator";
  inputs.vstat.url = "github:heal-research/vstat/main";

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
            "-DBUILD_SHARED_LIBS=${if pkgs.hostPlatform.isStatic then "OFF" else "ON"}"
            "-DBUILD_TESTING=OFF"
            "-DCMAKE_BUILD_TYPE=Release"
            "-DUSE_OPENLIBM=ON"
            "-DUSE_SINGLE_PRECISION=ON"
            "-DCMAKE_CXX_FLAGS=${if pkgs.hostPlatform.isx86_64 then "-march=x86-64-v3" else ""}"
          ];

          nativeBuildInputs = with pkgs; [ cmake ];

          buildInputs = with pkgs; [
            cxxopts
            doctest
            eigen
            fmt_8
            git
            openlibm
            pkg-config
            pratt-parser.defaultPackage.${system}
            vstat.packages.${system}.default
            # foolnotion overlay 
            aria-csv
            fast_float
            robin-hood-hashing
            scnlib
            span-lite
            taskflow
            vectorclass
            xxhash
          ];
        };

      in rec {
        packages.${system}.default = operon;
        defaultPackage = operon; 

        devShell = pkgs.stdenv.mkDerivation {
          name = "operon-env";
          hardeningDisable = [ "all" ];
          impureUseNativeOptimizations = true;
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
            valgrind
            jemalloc
            linuxPackages.perf
            graphviz
            seer
          ]);

          shellHook = ''
            LD_LIBRARY_PATH=${
              pkgs.lib.makeLibraryPath [ pkgs.stdenv.cc.cc.lib ]
            };
            alias bb="cmake --build build -j"
          '';
        };
      });
}
