let
  pkgs = import <nixpkgs> {
  };
  antigravity = pkgs.callPackage ./antigravity.nix {};

in pkgs.mkShell.override { stdenv = pkgs.clangStdenv; } {
  buildInputs = with pkgs; [
  ];
  packages = [
    antigravity
    pkgs.python313
  ] ++ (with pkgs.python313Packages; [
    matplotlib pandas numpy statsmodels yfinance 
    boto3 pyyaml zstandard
    ipython jupyter ipykernel pip
    mplfinance seaborn
  ]) ;
}
