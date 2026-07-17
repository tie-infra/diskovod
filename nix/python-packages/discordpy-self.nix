{
  aiohttp,
  audioop-lts,
  buildPythonPackage,
  curl-cffi,
  discord-protos,
  fetchFromGitHub,
  setuptools,
  tzlocal,
}:

buildPythonPackage rec {
  pname = "discord.py-self";
  version = "2.1.0";
  pyproject = true;

  src = fetchFromGitHub {
    owner = "dolfies";
    repo = "discord.py-self";
    rev = "v${version}";
    hash = "sha256-jVz3uGU+4E5Awbk6ZYAsXvEpClNHm2QN1RpBTIiQTpE=";
  };

  build-system = [ setuptools ];
  dependencies = [
    aiohttp
    audioop-lts
    curl-cffi
    discord-protos
    tzlocal
  ];

  pythonImportsCheck = [ "discord" ];
}
