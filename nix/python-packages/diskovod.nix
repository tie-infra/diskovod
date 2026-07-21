{
  buildPythonPackage,
  setuptools,
  aiosqlite,
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
  pydantic,
  python-multipart,
  regex,
  uvicorn,
  pytest,
  pytest-asyncio,
  pytestCheckHook,
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
    regex
    uvicorn
  ];

  nativeCheckInputs = [
    pytest
    pytest-asyncio
    pytestCheckHook
  ];

  pythonImportsCheck = [ "diskovod" ];
}
