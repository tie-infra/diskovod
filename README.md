# Diskovod

Diskovod is a local-first Discord DM assistant built on LangChain and LangGraph. It can use a
ChatGPT Plus/Pro subscription or a custom OpenAI-compatible Responses or Chat Completions endpoint.
The assistant participates openly as an AI rather than impersonating the account owner.

Each Discord chat has an independent encrypted LangGraph checkpoint thread. Diskovod keeps Discord
events and delivery outcomes as audit records, uses LangGraph Store for durable memories, and lets
LangChain's `create_agent` own the model/tool loop.

## Highlights

- Automatic, inline-collaboration, and paused modes per conversation.
- Live same-chat steering at safe agent boundaries; other chats remain isolated.
- Natural multi-message and progress replies through `send_messages`; the model can continue using
  tools afterward or explicitly finish after a successful sole send.
- Valid owner escalation through a LangGraph interrupt and dashboard resume. Invalid arguments use
  a localized fixed reply without another model request.
- Current date/time, bounded arithmetic, web search/fetch, attachment retrieval, reactions, and
  per-chat memory tools. There is deliberately no terminal-execution tool.
- Provider-hosted web search after a successful saved capability probe, otherwise bounded
  application-managed public-web access.
- Native image/file inputs according to saved model capabilities, plus content-addressed attachment
  storage and bounded lexical retrieval.
- A Bootstrap admin interface with Connections, Assistant, Conversations, Diagnostics, and Database
  tabs and System, Light, Dark, and OLED Black themes.
- Localized admin text, prompts, tool schemas, tool errors, and fixed replies in English, Russian,
  Ukrainian, Japanese, German, French, and Simplified Chinese.
- Local diagnostics for normalized model exchanges, provider probes, checkpoint history, safe
  replay, memories, attachments, delivery claims, Discord events, and live-injection batches.
- One SQLite database for relational state and encrypted checkpoints. Large attachment bodies are
  immutable content-addressed files covered by backup manifests.

## Configuration

Diskovod reads non-secret runtime configuration from JSON:

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

`public_url` is the browser-facing URL and OAuth callback base. The admin password must contain at
least 12 characters and the encryption secret at least 32 characters. Discord, OAuth, and provider
credentials entered in the admin UI are encrypted before persistence.

Create local secret files and start Diskovod:

```sh
install -m 600 /dev/null admin-password
install -m 600 /dev/null secret-key
printf '%s\n' 'replace-with-a-long-password' > admin-password
nix shell nixpkgs#openssl -c sh -c 'openssl rand -hex 32 > secret-key'
cp diskovod.example.json diskovod.json
nix run -- --config ./diskovod.json
```

Open `http://localhost:3090` and authenticate as `admin` with the password file's contents.

## Providers

### ChatGPT subscription

Select **Sign in with ChatGPT**. OpenAI redirects to its registered
`http://localhost:1455/auth/callback`. If Diskovod runs remotely, replace only that origin/path in
the browser address bar with `<public_url>/chatgpt/oauth/callback` and preserve the entire query
string. Diskovod exchanges the code with PKCE and stores rotating credentials encrypted.

The subscription adapter is isolated behind Diskovod's provider registry and returns a standard
LangChain `BaseChatModel`. It uses the reviewed `langchain-openai` subscription surface with zero
model retries. A runtime failure never switches model, provider, endpoint, or transport.

**Test hosted web search** executes a real server-tool request for the active model. Diagnostics keep
the exact normalized request, full normalized LangChain response/content blocks, response metadata,
status, and conclusion. The UI explicitly says when raw provider wire transport is unavailable; it
does not present cleaned metadata as a raw protocol trace. A failed or inconclusive probe leaves
hosted search disabled until an administrator reruns and saves a successful result.

### Custom OpenAI-compatible API

Enter a display name, an API base URL (normally ending in `/v1`), and an optional API key. New setup
preselects Responses. **Detect API support** probes the draft provider and preselects discovered
capabilities, but nothing is saved until the form is submitted. The selected protocol is pinned:
Diskovod never falls back to Chat Completions when Responses later fails.

Required client-owned tool calling and optional hosted search, native image/file input, and prompt
cache routing are saved as capability evidence. API effort values remain provider-defined stable
enums; only their UI labels are localized.

The runtime depends on `BaseChatModel`, standard messages/content blocks, and standard tool calls.
Additional providers belong in a dedicated integration adapter, credential schema, setup probe, and
registry entry; the agent loop and checkpoint schema do not change.

## Agent behavior

Ordinary final model text is internal and ends a run without appearing in Discord. Visible output
uses `send_messages`, which accepts one to five natural messages. It normally returns control to the
model, enabling this flow:

1. send a short progress message;
2. search or fetch a page;
3. use other tools as needed;
4. send one or more final messages;
5. stop naturally or set `continue_after_sending=false` on the sole successful final send.

A run may intentionally produce no Discord action. Diskovod does not parse emoji conventions,
force a named tool, or ask the model to repair malformed output. Tool call limits bound runaway
execution. External actions use a durable ledger keyed by run and tool-call IDs, so a completed call
is not deliberately executed again after restart.

Forced replies set a policy for the current run but do not force a tool choice; the model may
research before writing. Inline mode treats owner and peer messages as participant-labelled human
input and allows the assistant to help either participant or remain silent.

If someone asks Diskovod to run terminal commands, the prompt permits a playful textual Linux
terminal simulation. No command or filesystem tool is available to the model.

## Identity, personality, and localization

The default localized names are Diskovod, Дисковод (Russian and Ukrainian), ディスコヴォド,
Diskowod, Disquovode, and 迪斯科沃德. The admin may configure another name. Diskovod may honestly say
that it is an AI assistant. **Prefix generated replies with 🤖** adds a delivery-time marker without
changing checkpoint history; **Send without notifications** uses Discord's suppress-notifications
send option rather than visible `@silent` text.

The Personality cache can infer a reusable style profile from bounded, owner-authored Discord
history or pasted samples. Peer messages and generated replies are excluded. The inferred profile
can be reviewed and edited, and owner-provided details remain separate authoritative context.

The stable localized identity, personality, behavior, and tool-policy prefix precedes volatile
conversation context. When a provider advertises prompt-cache routing, Diskovod supplies a shared
hash derived from provider/model/transport, locale, identity, personality, owner details, and tool
schema—not the chat ID. This permits cross-chat prefix reuse without caching model responses or
tool actions.

## Persistence, memory, and attachments

`diskovod.sqlite3` contains configuration, Discord audit events, the per-chat event queue, delivery
claims, provider probes, local traces, LangGraph Store rows, attachment metadata, and encrypted
LangGraph checkpoints. Checkpoints are the authoritative short-term conversation state; Diskovod
does not reconstruct a bounded transcript for each invocation.

Long conversations are summarized by LangChain middleware. At the configured retention boundary,
Diskovod creates a new checkpoint generation seeded by a portable canonical summary. A provider,
model, or transport change also rolls completed conversations before the new configuration is used,
so provider-affine state is not replayed into an incompatible provider.

Incoming attachment bodies are addressed by SHA-256 under `attachments/`. Supported native content
is included only according to the saved capability profile. Extracted text is bounded, indexed, and
clearly marked as untrusted conversation data. Per-chat memory tools can remember, search, and
forget explicit facts with provenance; memory never becomes privileged prompt text merely because
it was retrieved.

## Diagnostics and administration

The Diagnostics tab correlates runs with normalized model requests/responses, tool calls, errors,
and Discord delivery records. It also exposes:

- provider capability probes and their evidence;
- each chat thread and encrypted checkpoint history;
- historical replay using isolated in-memory Store state and emulated Discord actions;
- long-term memories and confirmed deletion;
- attachment ingestion records;
- side-effect claims/results;
- ordered Discord ingress, queue disposition, logical request, and injection batch.

Historical replay cannot send Discord messages, reactions, or real escalation acknowledgements.
The Database tab provides redacted, paginated access to supported tables and allows deletion only
for explicitly mutable domain rows.

## One-time migration

On the first start after upgrading, Discord ingress remains stopped while Diskovod:

1. creates and integrity-checks a SQLite backup;
2. verifies every referenced attachment object's size and SHA-256 and writes a JSON manifest beside
   the backup;
3. converts Discord history to stable audit events and seeds one encrypted checkpoint per chat;
4. imports owner details into namespaced memory;
5. converts pending legacy escalations into actual LangGraph interrupts without resending their
   acknowledgements;
6. archives unverifiable legacy request/usage records;
7. validates event counts, checkpoint reachability/decryption, database integrity, and delivery
   ledger states;
8. drops replaced tables and records the completed cutover marker.

The new runtime contains no legacy model client, parser, repair loop, transcript reconstruction, or
fallback path.

## Discord connection

Paste the account token into the Discord card. Diskovod uses `discord.py-self`, stores the token
encrypted, and reconnects with bounded backoff. CAPTCHA challenges appear in the admin UI for manual
completion. Review Discord's terms and the rules applicable to your account before using a
self-client.

## NixOS

The flake exposes `pkgs.diskovod` and `nixosModules.default`:

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

The service uses a dynamic user and its systemd state directory. Relative state paths resolve under
`/var/lib/diskovod`; the service user must be able to read both secret files.

## Development

```sh
nix fmt
nix flake check
nix develop -c pytest -q
nix build
```
