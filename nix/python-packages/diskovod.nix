{
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
  httpcore2,
  httpx2,
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
    aiosqlite
    cryptography
    discordpy-self
    fastapi
    jinja2
    langchain
    langchain-openai
    langgraph
    langgraph-checkpoint-sqlite
    httpcore2
    httpx2
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
