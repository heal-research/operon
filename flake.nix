{
  description = "Operon development environment";

  inputs = {
    flake-parts.url = "github:hercules-ci/flake-parts";
    nixpkgs.url = "github:nixos/nixpkgs/master";
    foolnotion.url = "github:foolnotion/nur-pkg";
    fluky.url = "github:foolnotion/fluky";
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
    fluky.inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs = inputs@{ self, flake-parts, nixpkgs, foolnotion, fluky, pratt-parser, vdt, vstat, lbfgs }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      systems = [ "x86_64-linux" "x86_64-darwin" "aarch64-linux" "aarch64-darwin" ];

      perSystem = { pkgs, system, ... }:
        let
          pkgs = import self.inputs.nixpkgs {
            inherit system;
            overlays = [
              foolnotion.overlay
              (final: prev: {
                fluky = fluky.packages.${system}.default;
                lbfgs = lbfgs.packages.${system}.default;
                pratt-parser = pratt-parser.packages.${system}.default;
                vdt   = vdt.packages.${system}.default;
                vstat = vstat.packages.${system}.default;
              })
            ];
          };
          enableShared = !pkgs.stdenv.hostPlatform.isStatic;
         enableTesting = false;
          stdenv = pkgs.llvmPackages_20.stdenv;
          operon = import ./operon.nix { inherit stdenv pkgs system; enableShared = enableShared; enableTesting = enableTesting; };
        in
        rec
        {
          packages = {
            default = operon.overrideAttrs (old: {
              cmakeFlags = old.cmakeFlags ++ [ "-DBUILD_CLI_PROGRAMS=ON" "-DCPM_USE_LOCAL_PACKAGES=ON" ];
            });

            cli = operon.overrideAttrs (old: {
              cmakeFlags = old.cmakeFlags ++ [ "-DBUILD_CLI_PROGRAMS=ON" "-DCPM_USE_LOCAL_PACKAGES=ON" ];
            });

            cli-static = operon.overrideAttrs (old: {
              cmakeFlags = old.cmakeFlags ++ [ "-DBUILD_CLI_PROGRAMS=ON" "-DCPM_USE_LOCAL_PACKAGES=ON" ];
            });

            library = operon.overrideAttrs (old: {
              enableShared = true;
            });

            library-static = operon.overrideAttrs (old: {
              enableShared = false;
            });
          };

          devShells.default = stdenv.mkDerivation {
            name = "operon";

            nativeBuildInputs = operon.nativeBuildInputs ++ (with pkgs; [
              clang-tools
              cppcheck
              include-what-you-use
              cmake-language-server
            ]);

            buildInputs = operon.buildInputs ++ (with pkgs; [
              gdb
              gcc14
              graphviz
              hyperfine
              linuxPackages_latest.perf
              seer
              valgrind
              hotspot
            ]);
          };

          apps.operon-gp.program = "${packages.default}/bin/operon_gp";
          apps.operon-nsgp.program = "${packages.default}/bin/operon_nsgp";
          apps.operon-parse-model.program = "${packages.default}/bin/operon_parse_model";
        };
    };
}
