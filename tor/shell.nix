let
  pkgs = import <nixpkgs> { config={allowUnfree=true; }; };
in pkgs.mkShell { 
  packages = [ 
    pkgs.antigravity 
    pkgs.killall

    # Plugin development
    pkgs.python313
    ] ++ (with pkgs.python313.pkgs; [
      matplotlib                                                                
      pandas                                                                    
      numpy                                                                     
      ipython
      jupyter
      seaborn
      scikit-learn
      xgboost
      kmapper
      
      networkx
  ]); 
}
