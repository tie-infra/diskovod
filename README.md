# Diskovod

Diskovod is a Discord DM assistant backed by either a ChatGPT Plus/Pro subscription or a custom
OpenAI-compatible API. It watches DMs, replies using a cached personality profile, and provides an
admin UI for Discord, model providers, personality, and conversation controls.

When the account owner sends a message, Diskovod cancels pending work and enters a configurable
quiet window. Messages received during that window are recorded but not answered. Automation can
resume on the next incoming message after the window expires. Individual conversations can also be
paused until an administrator resumes them.

## Features

- ChatGPT OAuth with PKCE, refresh-token rotation, and subscription-backed streaming responses.
- Custom OpenAI-compatible Chat Completions providers, including keyless local endpoints.
- Discord connection through `discord.py-self`.
- Responsive Bootstrap admin navigation with a script-free interface.
- A guarded SQLite explorer with secret redaction, search, pagination, and confirmed row deletion.
- Personality inference from bounded Discord or pasted message history, with an editable cache.
- Editable owner details for names, preferences, relationships, plans, and other personal context.
- Transparent AI-assistant identity with an optional visible robot-emoji marker.
- Rare emoji reactions for lightweight acknowledgements when a written reply is unnecessary.
- Model-composed multi-message replies with configurable frequency, count, and timing.
- Optional Discord suppress-notifications flag for generated replies.
- Capability-aware attachment context with native image/file inputs and bounded text retrieval.
- Configurable generation caps for concise DM replies.
- Edit-aware message history that refreshes a pending reply when its trigger changes.
- Configurable opt-in or opt-out default with per-conversation enrollment controls.
- Detailed model token accounting by time window, model, and operation.
- SQLite storage with encrypted Discord, ChatGPT, and custom-provider credentials.
- An IPv6 loopback listener default, with a separate browser-facing public URL.
- Nix package overlay and a settings-based NixOS module.

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

The Discord token and custom-provider API key entered in the admin UI, along with ChatGPT
credentials, are encrypted before they are persisted in SQLite.

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

Select **Sign in with ChatGPT**. The OAuth client is registered only for
`http://localhost:1455/auth/callback`, so OpenAI redirects the browser there after sign-in. Because
Diskovod is remote, that page normally fails to load. In the browser address bar, replace only
`http://localhost:1455/auth/callback` with `<public_url>/chatgpt/oauth/callback`, preserving the
entire `?code=...&state=...` query string. Diskovod then exchanges the code using the original,
registered localhost callback and redirects to the admin UI.

### OpenAI-compatible provider

In **OpenAI-compatible API**, enter a display name, an API base URL that normally ends in `/v1`,
and an optional API key. Saving the provider also selects it. Diskovod makes non-streaming requests
to `<base_url>/chat/completions` with a system message followed by the DM history. The model name is
configured separately in **Reply behavior**.

API keys use the standard `Authorization: Bearer` header. Leaving the key empty supports local or
otherwise keyless endpoints; the example URL is `http://localhost:8000/v1`. The custom transport
intentionally uses the broadly compatible Chat Completions request shape. Reasoning effort remains
specific to the ChatGPT Subscription transport.

### Discord sign-in

Paste the account token into the Discord card in the admin UI. Diskovod stores it encrypted and
reconnects immediately using
[`discord.py-self`'s manual-token authentication flow](https://discordpy-self.readthedocs.io/en/latest/authenticating.html)
and default Discord endpoints.

Discord login and connection failures do not stop the service. Diskovod recreates the client with
bounded exponential backoff and clears the displayed connection error after reconnecting. ChatGPT
and custom model calls use request-scoped connections, so a transient failure affects that
operation while later calls can try again.

### CAPTCHA requests

When `discord.py-self` encounters a CAPTCHA, Diskovod holds the triggering request open for up to
ten minutes and shows the challenge metadata in the Discord card. Solve the challenge using its
service, site key, API page, and optional request data, then paste the returned solution token into
the pending request form. The library retries the original request with that solution.

## Personality preload

In **Personality cache**, choose how many recent Discord messages to inspect or paste a message
history manually. The Discord loader sends only the account's human-authored messages to the active
model provider and excludes generated replies. Identical history reuses the cached profile, and the
raw preload text is not retained.

The Discord loader inspects between 20 and 500 recent messages across the most recently active DMs,
with an 80,000-character inference limit. It labels conversations anonymously and marks whether an
owner message stands alone or immediately continues an owner-authored message burst; peer content is
not included in these annotations.

The inferred description gives special attention to the owner's dominant message shape: typical
length and line count, sentence fragments, punctuation, and the frequency and density of lists. It
also covers consecutive-message frequency, typical burst size and boundaries, communication habits,
preferences, languages, recurring interests, social style, temperament, and other supported traits.
Rare behavior is labeled as contextual instead of being treated as the default. The cached
description can be reviewed and edited in the admin UI at any time. Re-running inference after a
prompt revision refreshes the cache even when the selected history has not changed.

The profile ends with 8–12 synthetic representative examples written from the inferred style.
They are newly generated examples—not samples, quotations, or close paraphrases from the private
history. This gives the reply model concrete style guidance without copying source messages.

**Owner details** in Reply behavior provides facts that message history may not express clearly,
such as the owner's name, location, work, interests, relationships, preferences, or recurring plans.
These details are treated as authoritative when they conflict with an inferred trait. The reply
prompt directs the model to use them only when relevant and not to volunteer unrelated personal or
sensitive information.

The default reply instructions identify Diskovod as an AI assistant helping the account owner,
rather than instructing it to impersonate the owner. If asked about its identity or a reply's
origin, it answers honestly. Installations that still have the exact former default saved migrate
to the transparent prompt automatically; custom instructions are never rewritten.

For lightweight acknowledgements, Diskovod may react to the incoming message with one common emoji
instead of sending text. Reactions are never combined with a reply. A local limiter permits at most
one reaction among the latest twelve automated actions and applies a six-hour per-conversation
cooldown; if the model proposes one sooner, it is asked for a normal text reply instead.

**Multi-message replies** lets the model occasionally compose a sequence of two to five distinct
Discord messages. The admin UI controls whether sequences are available, the percentage of turns
on which they are offered, the maximum message count, and the randomized delay between messages.
Diskovod does not split completed prose mechanically: the model chooses the boundaries and may keep
a single message when a sequence would feel forced. Before every part, automation and recent manual
owner activity are checked again, so the remainder stops if the owner joins the conversation.

**Send generated replies without notifications** uses Discord's suppress-notifications message
flag. It does not modify the visible message text and does not affect reactions.

**Prefix generated replies with 🤖** adds a visible marker to every generated text message so the
recipient can distinguish assistant replies from messages written manually by the account owner.
The marker is not stored in conversation history and does not affect reactions.

Incoming messages retain metadata for up to four attachments. Supported images are passed as
native vision inputs to documented vision-capable model families. With the ChatGPT Subscription
transport, supported documents up to 20 MiB are passed as Responses `input_file` URLs. Custom
Chat Completions providers receive the broadly compatible image-URL format, but not native document
parts. For either transport, Diskovod downloads up to 64 KiB total from small text/code attachments
when each message arrives and adds up to 24,000 characters per file to the prompt as bounded
retrieval context. Unsupported and oversized files still contribute filename, media type, size,
and description metadata, without downloading their bodies.
Native image and document URLs are sent only for the message that triggered the current reply;
later turns retain their metadata and captured text without replaying expiring Discord CDN URLs.

**Reply token budget** applies to each DM generation and any repair or reaction-fallback generation.
The ChatGPT Subscription transport rejects `max_output_tokens`, so Diskovod expresses its value as
a best-effort length instruction instead. Custom providers receive a hard `max_completion_tokens`
limit. Personality inference uses a separate 2,000-token budget so its profile and examples are not
truncated by the concise reply setting.

## Token usage

Diskovod orders reusable base instructions, owner details, the cached personality, and reply-safety
rules before changing manual-message examples and conversation history. This preserves the longest
possible exact prompt prefix for automatic provider caching. ChatGPT Subscription requests also use
a stable, hashed `prompt_cache_key` per model and DM so growing conversation prefixes are routed
consistently without exposing Discord channel IDs. Personality inference uses a separate stable key.
Custom providers receive the cache-friendly prompt order but no cache-specific request fields, since
not every OpenAI-compatible server accepts them.

Diskovod records the usage metadata reported with each completed response: input tokens, cached
input tokens, output tokens, reasoning tokens, and total tokens. Both ChatGPT Subscription and
custom Chat Completions usage formats are normalized into the same counters. The admin UI shows
rolling 24-hour, 7-day, 30-day, and all-time totals; breakdowns by model and operation; cache
utilization; and the 50 most recent calls.

Usage rows contain only the response ID, timestamp, model, operation, and token counters. They do
not duplicate prompts or generated messages. Calls made before this feature was enabled cannot be
reconstructed, and a completed response that omits usage metadata is not estimated.

## Human activity

The **New conversation default** determines enrollment when Diskovod first observes a DM. With
**Opt in**, new conversations may be automated immediately. With **Opt out**, their messages are
recorded but no reply is generated until automation is enabled for that conversation in the admin
UI. Changing the default does not alter conversations already known to Diskovod.

Diskovod records a nonce before each generated send. A self-authored Gateway event without that
nonce is treated as human activity: active generation is cancelled and a random quiet window
starts. The task checks the current generation and pause state before typing and again before
sending. It also checks recent Discord history immediately before sending in case the Gateway
event is delayed.

Messages received during a quiet window are not queued for a later reply. Conversation enrollment
is separate from the quiet window and remains disabled until **Enable automation** is selected.

Message content edits are reflected in stored history. Editing an outgoing message marks its final
version as human-authored, cancels pending automation, and starts the normal quiet window. If the
peer edits the exact incoming message that currently has a reply pending, Diskovod cancels and
regenerates from the edited content. Older edits update context but do not cause duplicate replies.

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
