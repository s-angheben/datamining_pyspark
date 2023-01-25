# shell.nix
{ pkgs ? import <nixpkgs> {} }:
let
  my-pythonPackages = p: with p; [
    pandas
    torch
    pyspark
    pyarrow

    jupyter
  ];
  my-python = pkgs.python3.withPackages my-pythonPackages;
in
pkgs.mkShell {
  packages = [
    my-python

#    pkgs.spark
    pkgs.jdk
  ];

  # shellHook = "jupyter notebook";
}
