# Diskovod

Diskovod is a Discord DM assistant backed by a ChatGPT Plus or Pro subscription. It watches DMs,
replies using a cached personality profile, and provides an admin UI for Discord, ChatGPT,
personality, and conversation controls.

When the account owner sends a message, Diskovod cancels pending work and enters a configurable
quiet window. Messages received during that window are recorded but not answered. Automation can
resume on the next incoming message after the window expires. Individual conversations can also be
paused until an administrator resumes them.

## Features

- ChatGPT OAuth with PKCE, refresh-token rotation, and subscription-backed streaming responses.
- Discord connection through `discord.py-self`.
- Personality inference from bounded Discord or pasted message history, with an editable cache.
- Detailed ChatGPT token accounting by time window, model, and operation.
- SQLite storage with encrypted Discord and ChatGPT credentials.
- An IPv6 loopback listener default, with a separate browser-facing public URL.
- Nix package overlay and a settings-based NixOS module.

The ChatGPT transport follows Zed's
[`openai_subscribed` provider](https://github.com/zed-industries/zed/blob/master/crates/language_models/src/provider/openai_subscribed.rs).

## Configuration

Non-secret configuration is read from JSON:

```json
{
  "host": "::1",
  "port": 3090,
  "public_url": "http://localhost:3090",
  "data_dir": "./data",
  "log_level": "INFO",
  "admin_password_file": "./admin-password",
  "secret_key_file": "./secret-key"
}
```

`host` is the listen address. `public_url` is the URL opened in the browser and is used for the
ChatGPT callback. For a remote deployment, set `host` to an appropriate IPv6 listen address and
`public_url` to the HTTPS URL served by the reverse proxy.

Secrets are accepted through file paths in the JSON settings:

| Setting | File contents | Required |
| --- | --- | --- |
| `admin_password_file` | Admin UI password, at least 12 characters | yes |
| `secret_key_file` | Database encryption key, at least 32 characters | yes |

The Discord token entered in the admin UI and ChatGPT credentials are encrypted before they are
persisted in SQLite.

## Run

Create protected secret files:

```sh
install -m 600 /dev/null admin-password
install -m 600 /dev/null secret-key
printf '%s\n' 'replace-with-a-long-password' > admin-password
nix shell nixpkgs#openssl -c sh -c 'openssl rand -hex 32 > secret-key'
```

Then run:

```sh
cp diskovod.example.json diskovod.json
nix run -- --config ./diskovod.json
```

Open `http://localhost:3090` and authenticate as `admin` using the password file's contents.

### ChatGPT sign-in

Select **Sign in with ChatGPT**. The authorization request returns to
`<public_url>/chatgpt/oauth/callback`, then redirects to the admin UI.

### Discord sign-in

Paste the account token into the Discord card in the admin UI. Diskovod stores it encrypted and
reconnects immediately using
[`discord.py-self`'s manual-token authentication flow](https://discordpy-self.readthedocs.io/en/latest/authenticating.html)
and default Discord endpoints.

Discord login and connection failures do not stop the service. Diskovod recreates the client with
bounded exponential backoff and clears the displayed connection error after reconnecting. ChatGPT
uses request-scoped connections, so a transient failure affects that operation while later calls
can try again.

### CAPTCHA requests

When `discord.py-self` encounters a CAPTCHA, Diskovod holds the triggering request open for up to
ten minutes and shows the challenge metadata in the Discord card. Solve the challenge using its
service, site key, API page, and optional request data, then paste the returned solution token into
the pending request form. The library retries the original request with that solution.

## Personality preload

In **Personality cache**, choose how many recent Discord messages to inspect or paste a message
history manually. The Discord loader sends only the account's human-authored messages to ChatGPT
and excludes generated replies. Identical history reuses the cached profile, and the raw preload
text is not retained.

The Discord loader inspects between 20 and 500 recent messages across the most recently active DMs,
with an 80,000-character inference limit.

The inferred description covers communication habits, preferences, languages, recurring
interests, social style, temperament, and other traits supported by the messages. The cached
description can be reviewed and edited in the admin UI at any time.

## Token usage

Diskovod records the usage metadata reported with each completed ChatGPT response: input tokens,
cached input tokens, output tokens, reasoning tokens, and total tokens. The admin UI shows rolling
24-hour, 7-day, 30-day, and all-time totals; breakdowns by model and operation; cache utilization;
and the 50 most recent calls.

Usage rows contain only the response ID, timestamp, model, operation, and token counters. They do
not duplicate prompts or generated messages. Calls made before this feature was enabled cannot be
reconstructed, and a completed response that omits usage metadata is not estimated.

## Human activity

Diskovod records a nonce before each generated send. A self-authored Gateway event without that
nonce is treated as human activity: active generation is cancelled and a random quiet window
starts. The task checks the current generation and pause state before typing and again before
sending. It also checks recent Discord history immediately before sending in case the Gateway
event is delayed.

Messages received during a quiet window are not queued for a later reply. Permanent pause is a
separate per-conversation setting and remains active until **Resume now** is selected.

## Overlay

The default overlay exposes `pkgs.diskovod`:

```nix
{ inputs, pkgs, ... }:
{
  nixpkgs.overlays = [ inputs.diskovod.overlays.default ];
  environment.systemPackages = [ pkgs.diskovod ];
}
```

## NixOS module

Add the overlay, import `inputs.diskovod.nixosModules.default`, and put the complete runtime
configuration under `services.diskovod.settings`. The module renders the settings to JSON.

```nix
{ inputs, ... }:
{
  nixpkgs.overlays = [ inputs.diskovod.overlays.default ];
  imports = [ inputs.diskovod.nixosModules.default ];

  services.diskovod = {
    enable = true;
    settings = {
      host = "::";
      port = 3090;
      public_url = "https://diskovod.internal.example";
      log_level = "INFO";
      admin_password_file = "/run/secrets/diskovod-admin-password";
      secret_key_file = "/run/secrets/diskovod-secret-key";
    };
  };
}
```

The service uses a dynamic system user and runs from its systemd state directory. Relative state
paths therefore resolve beneath `/var/lib/diskovod`. The dynamic user must be able to read each
secret file. Use an IPv6-capable TLS reverse proxy when exposing the admin UI beyond loopback.

## Development

```sh
nix fmt
nix flake check
nix develop -c pytest -q
nix build
```
