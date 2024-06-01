{
  description = "Operon development environment";

  inputs = {
    flake-parts.url = "github:hercules-ci/flake-parts";
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

  outputs = inputs@{ self, flake-parts, nixpkgs, foolnotion, pratt-parser, vdt, vstat, lbfgs }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      systems = [ "x86_64-linux" "x86_64-darwin" "aarch64-darwin" ];

      perSystem = { pkgs, system, ... }:
        let
          pkgs = import self.inputs.nixpkgs {
            inherit system;
            overlays = [
              foolnotion.overlay
              (final: prev: {
                vdt   = vdt.packages.${system}.default;
                vstat = vstat.packages.${system}.default;
                lbfgs = lbfgs.packages.${system}.default;
                pratt-parser = pratt-parser.packages.${system}.default;
              })
            ];
          };
          stdenv = pkgs.llvmPackages_18.stdenv;
          operon = import ./operon.nix { inherit stdenv pkgs system; };
        in
        rec
        {
          packages = {
            default = operon.overrideAttrs (old: {
              cmakeFlags = old.cmakeFlags ++ [ "-DBUILD_CLI_PROGRAMS=ON" ];
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
              clang-tools_18
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
          };

          apps.operon-gp.program = "${packages.default}/bin/operon_gp";
          apps.operon-nsgp.program = "${packages.default}/bin/operon_nsgp";
          apps.operon-parse-model.program = "${packages.default}/bin/operon_parse_model";
        };
    };
}
