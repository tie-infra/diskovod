<p align="center">
  <img src="docs/logo.svg" width="128" height="128" alt="Diskovod logo">
</p>

<h1 align="center">Diskovod</h1>

<p align="center">A local-first, human-aware Discord DM assistant.</p>

Diskovod works openly alongside the account owner: it can answer automatically, help both people
inline, or stay out of the way. It uses LangChain and LangGraph for agent execution, tools, memory,
steering, and persistent per-chat conversations.

> [!IMPORTANT]
> Diskovod connects through `discord.py-self`. Review Discord's terms and the rules applicable to
> your account before using a self-client.

## Highlights

- Independent encrypted LangGraph conversation state, durable memory, and automation mode for each
  Discord chat, with new messages injected at safe agent boundaries.
- Natural multi-message and progress replies, reactions, suppressed notifications, optional robot
  markers, and dashboard-managed owner escalation.
- ChatGPT subscription sign-in and custom OpenAI-compatible Responses or Chat Completions
  providers behind a provider-neutral LangChain model interface.
- Current date and time, arithmetic, public web search and fetch, Discord attachment retrieval, and
  explicit per-chat memory tools—without terminal execution access.
- Shared HTTP/2 client for untrusted URLs with transport-level public-address enforcement, redirect
  protection, bounded downloads, connection reuse, and safe proxy handling.
- Bootstrap administration with conversation controls, provider setup and probes, diagnostics,
  checkpoint inspection, data management, and System, Light, Dark, and OLED Black themes.
- English, Russian, Ukrainian, Japanese, German, French, and Simplified Chinese UI, prompts, tool
  schemas, errors, and fixed replies in a machine-editable JSON catalog.
- One local SQLite database for application state and encrypted checkpoints, plus content-addressed
  attachment storage and correlated model/tool/delivery traces.

## Development

```sh
nix fmt
nix flake check
nix develop -c pytest -q
nix build
```

## JSON configuration

Diskovod reads non-secret runtime settings from a JSON file and references secrets by path. Discord,
OAuth, and provider credentials entered in the admin UI are encrypted before persistence.

```json
{
  "host": "::1",
  "port": 3090,
  "public_url": "http://localhost:3090",
  "data_dir": "./data",
  "log_level": "INFO",
  "log_levels": {
    "uvicorn.access": "WARNING"
  },
  "admin_password_file": "./admin-password",
  "secret_key_file": "./secret-key"
}
```

`log_level` sets the default logging threshold. `log_levels` overrides it for individual Python
logger namespaces; names are hierarchical, so `diskovod` controls all application loggers while a
more specific entry such as `diskovod.runtime` controls only that component. Supported levels are
`DEBUG`, `INFO`, `WARNING`, `ERROR`, and `CRITICAL` (case-insensitive). Successful and unsuccessful
Uvicorn requests are all emitted at `INFO`, so the default `uvicorn.access` level of `WARNING`
disables access lines entirely. To troubleshoot Uvicorn without enabling other components or noisy
request logs, use:

```json
"log_levels": {
  "uvicorn": "DEBUG",
  "uvicorn.access": "WARNING"
}
```

The admin password must contain at least 12 characters and the encryption key at least 32. Start
from the included example:

```sh
cp diskovod.example.json diskovod.json
printf '%s\n' 'replace-with-a-long-password' > admin-password
nix shell nixpkgs#openssl -c sh -c 'openssl rand -hex 32 > secret-key'
chmod 600 admin-password secret-key
nix run -- --config ./diskovod.json
```

Open `http://localhost:3090`, sign in as `admin`, and complete Discord and model setup in the
dashboard. `public_url` must match the browser-facing origin used for OAuth callbacks.

## NixOS

The flake exposes `overlays.default`, `packages.diskovod`, and `nixosModules.default`:

```nix
{ inputs, ... }:
{
  imports = [ inputs.diskovod.nixosModules.default ];
  nixpkgs.overlays = [ inputs.diskovod.overlays.default ];

  services.diskovod = {
    enable = true;
    settings = {
      host = "::";
      port = 3090;
      public_url = "https://diskovod.example.com";
      log_level = "INFO";
      log_levels."uvicorn.access" = "WARNING";
      admin_password_file = "/run/credentials/diskovod.service/admin-password";
      secret_key_file = "/run/credentials/diskovod.service/secret-key";
    };
  };

  systemd.services.diskovod.serviceConfig.LoadCredential = [
    "admin-password:/run/secrets/diskovod-admin-password"
    "secret-key:/run/secrets/diskovod-secret-key"
  ];
}
```

The module runs Diskovod under a dynamic user and stores relative application data beneath
`/var/lib/diskovod`. The credential sources in this example can be supplied by the host's secret
manager.
