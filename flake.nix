{
  description = "Rényi Differentially Private Federated ADMM";

  inputs.nixpkgs.url = "nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs, ... }:
  let
    system = "x86_64-linux";
    pkgs = nixpkgs.legacyPackages.${system};
    pypkgs = pkgs.python3Packages;
  in {
    packages.${system}.default = pypkgs.buildPythonApplication {
      pname = "rdp-fed-admm";
      version = "0.0.1";
      pyproject = true;
      doCheck = true;
      src = self;

      build-system = with pypkgs; [
        hatchling
      ];

      dependencies = with pypkgs; [
        numpy
      ];

      nativeCheckInputs = with pkgs; [
        ruff
        basedpyright
      ];

      checkPhase = ''
        ruff check --preview
        basedpyright
        basedpyright --ignoreexternal --verifytypes rdp_fed_admm
      '';

      meta = with pkgs.lib; {
        description = "Rényi Differentially Private Federated ADMM";
        license = licenses.gpl3;
        platforms = platforms.linux;
        mainProgram = "rdp-fed-admm";
      };
    };
  };
}
