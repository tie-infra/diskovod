# Diskovod

Diskovod is a Discord DM assistant backed by either a ChatGPT Plus/Pro subscription or a custom
OpenAI-compatible API. It watches DMs, replies using a cached personality profile, and provides an
admin UI for Discord, model providers, personality, and conversation controls.

In Automatic mode, an account-owner message cancels pending work and starts a configurable quiet
window. Inline collaboration instead lets the assistant participate openly alongside the owner,
help either participant when useful, and remain silent when it has nothing useful to add. Individual
conversations can also be paused.

## Features

- ChatGPT OAuth with PKCE, refresh-token rotation, and subscription-backed streaming responses.
- Custom OpenAI-compatible Responses or Chat Completions providers, including keyless local endpoints.
- Discord connection through `discord.py-self`.
- Server-rendered Bootstrap admin tabs with a system-color-scheme default and explicit Light, Dark,
  and OLED Black themes.
- A guarded SQLite explorer with secret redaction, search, pagination, and confirmed row deletion.
- Personality inference from bounded Discord or pasted message history, with an editable cache.
- Editable owner details for names, preferences, relationships, plans, and other personal context.
- Transparent AI-assistant identity with a localized or custom name and an optional visible
  robot-emoji marker.
- Rare emoji reactions for lightweight acknowledgements when a written reply is unnecessary.
- Model-composed multi-message replies with configurable count and timing.
- Optional Discord suppress-notifications flag for generated replies.
- Capability-aware attachment context with native image/file inputs and bounded text retrieval.
- Configurable generation caps for concise DM replies.
- Edit-aware message history that refreshes a pending reply when its trigger changes.
- One-shot forced written replies that can bypass automation enrollment and quiet windows.
- Dashboard owner escalation that acknowledges explicit requests and pauses the conversation.
- On-demand current date/time and bounded arithmetic tools.
- Capability-gated hosted web search with natural source links in replies.
- Configurable opt-in or opt-out default with per-conversation enrollment controls.
- Per-conversation Automatic, Inline collaboration, and Paused modes.
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

Hosted web search is not assumed to work merely because the subscription transport is
Responses-shaped. Use **Test hosted web search** for the selected model; the result is scoped to that
model and account and enables search only after the model completes both a real hosted search and a
terminal function call in the same probe.

The probe distinguishes verified support from request errors and response-contract mismatches.
**Probe diagnostics** shows the model, stable reasoning-effort value, transport, timestamp, provider
response ID, and sanitized tool-call counts and statuses. The same metadata is logged for
correlation. Search results, prompts beyond the fixed public test, credentials, and raw response
content are not retained. An inconclusive result does not by itself prove that the model lacks web
search; it can also indicate backend rollout, account access, or an integration issue.

### OpenAI-compatible provider

In **OpenAI-compatible API**, enter a display name, an API base URL that normally ends in `/v1`,
and an optional API key. New providers default to the Responses API; existing configurations retain
Chat Completions until explicitly changed. **Detect API support** probes Responses first during
setup, preselects the detected protocol and capabilities, and never saves them until the
administrator explicitly submits the form. Production requests always use the saved protocol and
never retry through the other endpoint after an error. The model name is configured separately in
**Reply behavior**.

API keys use the standard `Authorization: Bearer` header. Leaving the key empty supports local or
otherwise keyless endpoints; the example URL is `http://localhost:8000/v1`. Automated replies
require standard native function calls. Optional capabilities such as strict schemas, cache routing,
parallel-tool control, and Responses hosted web search are configured or detected independently
from the HTTP protocol. Reasoning effort remains specific to the ChatGPT Subscription transport.

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

The default reply instructions identify the assistant by name as an AI helping the account owner,
rather than instructing it to impersonate the owner. If asked about its identity or a reply's
origin, it answers honestly. A blank **Assistant name** follows the prompt language automatically:
Diskovod in English, Дисковод in Russian and Ukrainian, ディスコヴォド in Japanese, 迪斯科沃德 in
Chinese, Diskowod in German, and Disquovode in French. The administrator can replace that localized
default with one installation-wide custom name. The identity instruction is kept separate from the
editable base prompt so a custom communication prompt does not accidentally remove it.

For lightweight acknowledgements, Diskovod may react to the incoming message with one common emoji
instead of sending text. Reactions are never combined with a reply. A local limiter permits at most
one reaction among the latest twelve automated actions and applies a six-hour per-conversation
cooldown; if the model proposes one sooner, one bounded native repair requests a written reply.

**Multi-message replies** lets the model compose a sequence of two to five distinct Discord
messages when the conversation naturally calls for one. The admin UI controls whether sequences are
available, the maximum message count, and the randomized delay between messages.
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
Responses providers receive native image inputs, while custom Chat Completions providers receive
the broadly compatible image-URL format; neither custom transport receives native document parts.
For every transport, Diskovod downloads up to 64 KiB total from small text/code attachments
when each message arrives and adds up to 24,000 characters per file to the prompt as bounded
retrieval context. Unsupported and oversized files still contribute filename, media type, size,
and description metadata, without downloading their bodies.
Native image and document URLs are sent only for the message that triggered the current reply;
later turns retain their metadata and captured text without replaying expiring Discord CDN URLs.

**Reply token budget** applies to each DM generation and any native repair or tool continuation.
The ChatGPT Subscription transport rejects `max_output_tokens`, so Diskovod expresses its value as
a best-effort length instruction instead. Custom providers receive a hard `max_completion_tokens`
limit. Personality inference uses a separate 2,000-token budget so its profile and examples are not
truncated by the concise reply setting.

## Native tools and owner escalation

Automated replies use native structured calls for sending one or more messages, reacting, or
escalating to the owner. Plain provider text is not sent directly. Diskovod validates message
counts, lengths, reactions, escalation state, and generation freshness before executing an action;
malformed output gets at most one bounded repair and otherwise fails closed.

The assistant can request the current date and time in the owner-configured IANA timezone and can
evaluate bounded arithmetic. These changing results exist only in the current model turn and are
never written to Discord history. A provider with verified Responses hosted-search support can also
search current public information using low search context. The final reply still goes through the
ordinary message action, may include a useful URL naturally, and is the only search-derived content
persisted locally. Raw search results and tool traces are not retained.

When a peer explicitly asks for the account owner, the model can write a context-aware
acknowledgement and create a dashboard escalation. Diskovod commits that record and permanently
pauses the conversation before sending the acknowledgement. Invalid escalation arguments use a
localized fixed acknowledgement without another model request. Claiming, resolving, dismissing, and
explicitly resuming are available in the dashboard. A manual owner reply resolves the active item
but leaves automation paused until the owner resumes it.

## Token usage

Diskovod orders reusable base instructions, owner details, the cached personality, and reply-safety
rules before changing manual-message examples and conversation history. This preserves the longest
possible exact prompt prefix for automatic provider caching. Chats with the same provider, model,
localized prompt, tool schema, owner details, and personality share a stable hashed
`prompt_cache_key`; their conversation histories remain separate request suffixes and are never
merged. Personality inference uses a separate stable key. Custom providers receive cache routing
only when that capability is enabled.

Diskovod records the usage metadata reported with each completed response: input tokens, cached
input tokens, output tokens, reasoning tokens, and total tokens. Both ChatGPT Subscription and
custom Responses and Chat Completions usage formats are normalized into the same counters. The admin UI shows
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

Each known conversation can then use one of three modes. **Automatic** replies to incoming messages
and yields to manual owner activity through the quiet window. **Inline collaboration** considers
messages from either participant, uses a dedicated no-op action when no contribution is needed, and
always prefixes generated contributions with 🤖. **Paused** records messages without generating.

Diskovod records a nonce before each generated send. A self-authored Gateway event without that
nonce is treated as human activity. In Automatic mode, active generation is cancelled and a random
quiet window starts. In Inline collaboration, the owner message becomes the newest shared turn and
may produce a clearly marked assistant follow-up. The task checks the current generation and pause
state before typing and again before sending. It also checks recent Discord history immediately
before sending in case the Gateway event is delayed.

Messages received during a quiet window are not queued for a later reply. Conversation enrollment
is separate from the quiet window and remains disabled until **Enable automation** is selected.

Message content edits are reflected in stored history. Editing an outgoing message marks its final
version as human-authored. It starts the normal quiet window in Automatic mode or becomes a revised
shared turn in Inline collaboration. If the peer edits the exact incoming message that currently has
a reply pending, Diskovod cancels and regenerates from the edited content. Older edits update context
but do not cause duplicate replies.

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
