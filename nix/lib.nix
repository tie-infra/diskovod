{ lib }:

lib
// {
  packagesFromDirectory =
    { directory, callPackage }:
    lib.packagesFromDirectoryRecursive {
      inherit directory callPackage;
    };
}
