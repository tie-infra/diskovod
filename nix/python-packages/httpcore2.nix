{
  anyio,
  buildPythonPackage,
  fetchPypi,
  h11,
  h2,
  hatch-fancy-pypi-readme,
  hatchling,
  socksio,
  truststore,
  uv-dynamic-versioning,
}:

buildPythonPackage rec {
  pname = "httpcore2";
  version = "2.7.0";
  pyproject = true;

  src = fetchPypi {
    inherit pname version;
    hash = "sha256-bcD+3zKaUqmQkwpVee3+uuqBEY6nAOoN194rXlvknvw=";
  };

  postPatch = ''
    substituteInPlace pyproject.toml \
      --replace-fail 'fallback-version = "0.0.0"' 'fallback-version = "${version}"'
  '';

  build-system = [
    hatch-fancy-pypi-readme
    hatchling
    uv-dynamic-versioning
  ];

  dependencies = [
    anyio
    h11
    h2
    socksio
    truststore
  ];

  pythonImportsCheck = [ "httpcore2" ];
}
