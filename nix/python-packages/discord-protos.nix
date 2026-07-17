{
  buildPythonPackage,
  fetchPypi,
  protobuf,
  setuptools,
}:

buildPythonPackage rec {
  pname = "discord-protos";
  version = "0.0.2";
  pyproject = true;

  src = fetchPypi {
    inherit pname version;
    hash = "sha256-I5U6BfMr7ttAtwjsS0V1MKYZaknI110zeukoKipByZc=";
  };

  build-system = [ setuptools ];
  dependencies = [ protobuf ];

  pythonImportsCheck = [ "discord_protos" ];
}
