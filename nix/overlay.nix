final: prev:
let
  pythonPackagesOverlay =
    python-final: _:
    final.lib.packagesFromDirectoryRecursive {
      inherit (python-final) callPackage newScope;
      directory = ./python-packages;
    };
in
{
  pythonPackagesExtensions = prev.pythonPackagesExtensions ++ [ pythonPackagesOverlay ];
  diskovod = final.python313.pkgs.toPythonApplication final.python313.pkgs.diskovod;
}
