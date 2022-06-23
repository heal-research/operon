{
  description = "Operon development environment";

  inputs.flake-utils.url = "github:numtide/flake-utils";
  inputs.nur.url = "github:nix-community/NUR";
  inputs.nixpkgs.url = "github:nixos/nixpkgs/master";
  inputs.pratt-parser.url = "github:foolnotion/pratt-parser-calculator";
  inputs.vstat.url = "github:heal-research/vstat/main";

  outputs = { self, flake-utils, nixpkgs, nur, pratt-parser, vstat }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          overlays = [ nur.overlay ];
        };
        repo = pkgs.nur.repos.foolnotion;

      in rec {
        defaultPackage = pkgs.gcc12Stdenv.mkDerivation {
          name = "operon";
          src = self;

          cmakeFlags = [
            "-DBUILD_CLI_PROGRAMS=ON"
            "-DBUILD_SHARED_LIBS=ON"
            "-DBUILD_TESTING=OFF"
            "-DCMAKE_BUILD_TYPE=Release"
            "-DUSE_OPENLIBM=ON"
            "-DUSE_SINGLE_PRECISION=ON"
            "-DCMAKE_CXX_FLAGS=${if pkgs.targetPlatform.isx86_64 then "-march=haswell" else ""}"
          ];

          nativeBuildInputs = with pkgs; [ cmake ];

          buildInputs = with pkgs; [
            ceres-solver
            cxxopts
            doctest
            eigen
            fmt
            git
            glog
            jemalloc
            openlibm
            pkg-config
            # flakes
            pratt-parser.defaultPackage.${system}
            vstat.defaultPackage.${system}
            # Some dependencies are provided by a NUR repo
            repo.aria-csv
            repo.fast_float
            repo.robin-hood-hashing
            repo.scnlib
            repo.span-lite
            repo.taskflow
            repo.vectorclass
            repo.xxhash
          ];
        };

        devShell = pkgs.gcc12Stdenv.mkDerivation {
          name = "operon-env";
          hardeningDisable = [ "all" ];
          impureUseNativeOptimizations = true;
          nativeBuildInputs = with pkgs; [
            bear
            cmake
            clang_14
            clang-tools
            cppcheck
            include-what-you-use
          ];
          buildInputs = defaultPackage.buildInputs ++ (with pkgs; [
            gdb
            hotspot
            valgrind
            jemalloc
            linuxPackages.perf
          ]);

          shellHook = ''
            LD_LIBRARY_PATH=${
              pkgs.lib.makeLibraryPath [ pkgs.gcc12Stdenv.cc.cc.lib ]
            };
          '';
        };
      });
}
