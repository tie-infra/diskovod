{
  description = "Diskovod — ChatGPT-subscription DM assistant";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-parts.url = "github:hercules-ci/flake-parts";
    treefmt-nix.url = "github:numtide/treefmt-nix";
  };

  outputs =
    inputs:
    let
      overlay = import ./nix/overlay.nix;
    in
    inputs.flake-parts.lib.mkFlake { inherit inputs; } {
      imports = [ inputs.treefmt-nix.flakeModule ];
      systems = [
        "aarch64-linux"
        "x86_64-linux"
      ];

      perSystem =
        { system, ... }:
        let
          pkgs = import inputs.nixpkgs {
            inherit system;
            overlays = [ overlay ];
          };
          python = pkgs.python313;
        in
        {
          packages = {
            inherit (pkgs) diskovod;
            default = pkgs.diskovod;
          };
          checks.diskovod = pkgs.diskovod;
          devShells.default = pkgs.mkShell {
            packages = [
              (python.withPackages (py: [
                py.diskovod
                py.pytest
                py.pytest-asyncio
              ]))
              pkgs.ruff
            ];
            shellHook = ''export PYTHONPATH="$PWD''${PYTHONPATH:+:$PYTHONPATH}"'';
          };

          treefmt = {
            projectRootFile = "flake.nix";
            programs = {
              nixfmt.enable = true;
              ruff-check.enable = true;
              ruff-format.enable = true;
            };
          };
        };

      flake = {
        overlays.default = overlay;
        nixosModules.default = ./nix/module.nix;
      };
    };
}
