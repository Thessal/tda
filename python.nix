let
  pkgs = import <nixpkgs> {
    config.allowUnfree = true;
    config.cudaSupport = false;
  };
  # spacyWordnet = pkgs.python3Packages.buildPythonPackage rec {
  #       pname = "tensorboardx";
  #       version = "0.0.5";
  # 
  #       src = pkgs.python3Packages.fetchPypi {
  #         inherit version;
  #         inherit pname;
  #         sha256 = "bErMjM0VIsQHnPFoEe80fPClpJyiCyNBuapegMgcPbc=";
  #       };
  # 
  #       propagatedBuildInputs = with pkgs.python3Packages; [ nltk spacy pyscaffold ];
  #       doCheck = false;
  #     };
  # stdenv = pkgs.gcc10Stdenv;
  pandarallel = pkgs.python3Packages.buildPythonPackage rec {
    pname = "pandarallel"; 
    version = "1.6.5"; # Must match the version in your setup.py/pyproject.toml
    pyproject = true;
    build-system = [ pkgs.python3Packages.setuptools ];
    src = pkgs.python3Packages.fetchPypi {
           inherit version;
           inherit pname;
           sha256 = "1c2df98ff6441e8ae13ff428ceebaa7ec42d731f7f972c41ce4fdef1d3adf640";
         };
         propagatedBuildInputs = with pkgs.python3Packages; [ 
		dill pandas psutil 
	 ];
         doCheck = false;
  };
  R-with-my-packages = pkgs.rWrapper.override{ packages = with pkgs.rPackages; [ ks ]; };
in pkgs.mkShell {
  packages = [
    pkgs.R
    #R-with-my-packages
    pandarallel
    (pkgs.python3.withPackages (python-pkgs: with python-pkgs; [
      # select Python packages here
      matplotlib
      pandas
      numpy
      ipython
      jupyter
      beautifulsoup4
      websockets
      pycryptodome
      kaggle
      mplfinance
      ipykernel
      seaborn
      virtualenv
      #torch
      python.pkgs.torchWithRocm

      #
      flask
      flask-socketio
      eventlet
      # tensorflow
      # tensorboard
      tensorboardx
      requests
      pycryptodomex 
      scikit-learn
      yfinance
      statsmodels

      zstandard
      polars
      orjson
      #swifter
      #rpy2
      #(rpy2.override {extraRPackages = with pkgs.rPackages; [ ks ];})
      
      ollama
    ]))
  ];
  # buildInputs = [ stdenv.cc.cc.lib ];
  # LD_LIBRARY_PATH = "${pkgs.stdenv.cc.cc.lib}/lib";
}
#{ pkgs ? import <nixpkgs> {} }:
#pkgs.mkShell {
  #buildInputs = with pkgs; [ rustc cargo rustfmt rustPackages.clippy ];
  #RUST_BACKTRACE = 1;
#}
