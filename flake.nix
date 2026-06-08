{
  description = "Development shell";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
  };

  outputs = { self, nixpkgs }:
  let
    system = "x86_64-linux";
    pkgs = import nixpkgs { inherit system; };
  in
  {
    devShells.x86_64-linux.default = pkgs.mkShell {
      packages = [
        (pkgs.python3.withPackages (python-pkgs: with python-pkgs; [
          opencv-python
          mss
          numpy
          pywinctl
          tensorflow
          matplotlib
        ]))
        (pkgs.retroarch.withCores (cores: with cores; [
          snes9x
        ]))
      ];
    };

    packages.x86_64-linux.hello = nixpkgs.legacyPackages.x86_64-linux.hello;

    packages.x86_64-linux.default = self.packages.x86_64-linux.hello;

  };
}
