{
  description = "Rust dev environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    let
      systems = with flake-utils.lib; [
        system."x86_64-linux"
        system."aarch64-linux"
        system."x86_64-darwin"
        system."aarch64-darwin"
      ];
    in 
      flake-utils.lib.eachSystem systems (system: 
        let
          pkgs = import nixpkgs {
            inherit system;
            config = {
              allowUnfree = true;
              allowUnfreePredicate = (_: true);
            };
          };
        in {
          devShells.default = pkgs.mkShell {
            buildInputs = with pkgs; [
                libcxx
                libiconv
            ];
          };
        });
}
