{
  aiohttp,
  aiosqlite,
  buildPythonPackage,
  cryptography,
  discordpy-self,
  fastapi,
  jinja2,
  langchain,
  langchain-openai,
  langgraph,
  langgraph-checkpoint-sqlite,
  pytest,
  pytest-asyncio,
  pydantic,
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
    aiosqlite
    cryptography
    discordpy-self
    fastapi
    jinja2
    langchain
    langchain-openai
    langgraph
    langgraph-checkpoint-sqlite
    pydantic
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
