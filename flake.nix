{
  description = "Operon development environment";

  inputs.flake-utils.url = "github:numtide/flake-utils";
  inputs.nur.url = "github:nix-community/NUR";
  inputs.nixpkgs.url = "github:nixos/nixpkgs/master";

  outputs = { self, flake-utils, nixpkgs, nur }:
    flake-utils.lib.eachDefaultSystem
      (system:
        let
          pkgs = import nixpkgs {
            inherit system;
            overlays = [ nur.overlay ];
          };
          repo = pkgs.nur.repos.foolnotion;
        in rec
        {
          defaultPackage = pkgs.gcc11Stdenv.mkDerivation {
            name = "operon";
            src = self;

            cmakeFlags = [ "-DCMAKE_BUILD_TYPE=Release" "-DUSE_SINGLE_PRECISION=ON" "-DBUILD_CLI_PROGRAMS=ON" "-DBUILD_TESTING=OFF" "-DBUILD_SHARED_LIBS=ON" ];

            nativeBuildInputs = with pkgs; [ cmake ];

            buildInputs = with pkgs; [
              ceres-solver
              cxxopts
              doctest
              eigen
              fmt
              git
              glog
              # Some dependencies are provided by a NUR repo
              repo.aria-csv
              repo.fast_float
              repo.pratt-parser
              repo.robin-hood-hashing
              repo.scnlib
              repo.span-lite
              repo.taskflow
              repo.vectorclass
              repo.vstat
              repo.xxhash
            ];
          };

          devShell = pkgs.gcc11Stdenv.mkDerivation {
            name = "operon-env";
            hardeningDisable = [ "all" ];
            impureUseNativeOptimizations = true;
            nativeBuildInputs = with pkgs; [ bear cmake clang_13 clang-tools cppcheck include-what-you-use ];
            buildInputs = defaultPackage.buildInputs ++ (with pkgs; [ gdb hotspot valgrind jemalloc linuxPackages.perf ]);

            shellHook = ''
              LD_LIBRARY_PATH=${pkgs.lib.makeLibraryPath [ pkgs.gcc11Stdenv.cc.cc.lib ]};
              '';
          };
        }
      );
}
