{
  aiohttp,
  buildPythonPackage,
  cryptography,
  discordpy-self,
  fastapi,
  jinja2,
  pytest,
  pytest-asyncio,
  python-multipart,
  setuptools,
  uvicorn,
}:

buildPythonPackage {
  pname = "diskovod";
  version = "0.1.0";
  pyproject = true;

  src = ../..;

  build-system = [ setuptools ];
  dependencies = [
    aiohttp
    cryptography
    discordpy-self
    fastapi
    jinja2
    python-multipart
    uvicorn
  ];

  nativeCheckInputs = [
    pytest
    pytest-asyncio
  ];
  checkPhase = "pytest -q";

  pythonImportsCheck = [ "diskovod" ];
}
