{
  anyio,
  buildPythonPackage,
  fetchPypi,
  h2,
  hatch-fancy-pypi-readme,
  hatchling,
  httpcore2,
  idna,
  truststore,
  uv-dynamic-versioning,
}:

buildPythonPackage rec {
  pname = "httpx2";
  version = "2.7.0";
  pyproject = true;

  src = fetchPypi {
    inherit pname version;
    hash = "sha256-izBwmu1chGWw3TuVwJzjAcj3nn56LQCrCvVR4NA3Wwc=";
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
    h2
    httpcore2
    idna
    truststore
  ];

  pythonImportsCheck = [ "httpx2" ];
}
