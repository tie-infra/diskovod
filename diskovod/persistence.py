from __future__ import annotations

import asyncio
import hashlib
import json
import os
from collections.abc import Iterable
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, AsyncIterator

import aiosqlite
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from langgraph.checkpoint.serde.encrypted import EncryptedSerializer
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.store.base import (
    BaseStore,
    GetOp,
    InvalidNamespaceError,
    Item,
    ListNamespacesOp,
    Op,
    PutOp,
    Result,
    SearchItem,
    SearchOp,
)


SQLITE_BUSY_TIMEOUT_MS = 5_000
TARGET_SCHEMA_VERSION = 14


TARGET_MIGRATIONS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS schema_migrations (
      version INTEGER PRIMARY KEY,
      applied_at REAL NOT NULL
    );
    CREATE TABLE IF NOT EXISTS agent_configuration_versions (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      created_at REAL NOT NULL,
      configuration TEXT NOT NULL,
      active INTEGER NOT NULL DEFAULT 0
    );
    CREATE UNIQUE INDEX IF NOT EXISTS one_active_agent_configuration
      ON agent_configuration_versions(active) WHERE active = 1;
    CREATE TABLE IF NOT EXISTS chat_threads (
      channel_id TEXT PRIMARY KEY,
      account_id TEXT NOT NULL,
      generation INTEGER NOT NULL DEFAULT 1,
      thread_id TEXT NOT NULL UNIQUE,
      live_steering INTEGER NOT NULL DEFAULT 1,
      queue_cursor INTEGER NOT NULL DEFAULT 0,
      updated_at REAL NOT NULL
    );
    CREATE TABLE IF NOT EXISTS discord_events (
      id TEXT PRIMARY KEY,
      channel_id TEXT NOT NULL,
      sequence INTEGER NOT NULL,
      kind TEXT NOT NULL,
      payload TEXT NOT NULL,
      observed_at REAL NOT NULL,
      UNIQUE(channel_id, sequence)
    );
    CREATE TABLE IF NOT EXISTS chat_event_queue (
      event_id TEXT PRIMARY KEY REFERENCES discord_events(id) ON DELETE CASCADE,
      channel_id TEXT NOT NULL,
      disposition TEXT NOT NULL DEFAULT 'pending',
      logical_request_id TEXT,
      injection_batch INTEGER,
      claimed_at REAL,
      completed_at REAL
    );
    CREATE INDEX IF NOT EXISTS chat_event_queue_ready
      ON chat_event_queue(channel_id, disposition, event_id);
    CREATE TABLE IF NOT EXISTS side_effect_deliveries (
      run_id TEXT NOT NULL,
      tool_call_id TEXT NOT NULL,
      action TEXT NOT NULL,
      state TEXT NOT NULL,
      request TEXT NOT NULL,
      result TEXT,
      claimed_at REAL NOT NULL,
      completed_at REAL,
      PRIMARY KEY(run_id, tool_call_id)
    );
    CREATE TABLE IF NOT EXISTS agent_runs (
      id TEXT PRIMARY KEY,
      thread_id TEXT NOT NULL,
      channel_id TEXT NOT NULL,
      trace_id TEXT NOT NULL UNIQUE,
      status TEXT NOT NULL,
      started_at REAL NOT NULL,
      completed_at REAL,
      error TEXT
    );
    CREATE TABLE IF NOT EXISTS agent_trace_events (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      run_id TEXT NOT NULL REFERENCES agent_runs(id) ON DELETE CASCADE,
      sequence INTEGER NOT NULL,
      kind TEXT NOT NULL,
      payload TEXT NOT NULL,
      recorded_at REAL NOT NULL,
      UNIQUE(run_id, sequence)
    );
    CREATE TABLE IF NOT EXISTS attachment_objects (
      sha256 TEXT PRIMARY KEY,
      size INTEGER NOT NULL,
      media_type TEXT,
      storage_path TEXT NOT NULL,
      created_at REAL NOT NULL
    );
    CREATE TABLE IF NOT EXISTS attachment_artifacts (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      object_sha256 TEXT NOT NULL REFERENCES attachment_objects(sha256) ON DELETE CASCADE,
      kind TEXT NOT NULL,
      state TEXT NOT NULL,
      content TEXT,
      metadata TEXT NOT NULL DEFAULT '{}',
      created_at REAL NOT NULL,
      updated_at REAL NOT NULL
    );
    CREATE TABLE IF NOT EXISTS attachment_chunks (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      artifact_id INTEGER NOT NULL REFERENCES attachment_artifacts(id) ON DELETE CASCADE,
      chunk_index INTEGER NOT NULL,
      content TEXT NOT NULL,
      metadata TEXT NOT NULL DEFAULT '{}',
      UNIQUE(artifact_id, chunk_index)
    );
    CREATE VIRTUAL TABLE IF NOT EXISTS attachment_chunks_fts USING fts5(
      content, content='attachment_chunks', content_rowid='id'
    );
    CREATE TABLE IF NOT EXISTS escalation_interrupts (
      id TEXT PRIMARY KEY,
      thread_id TEXT NOT NULL,
      channel_id TEXT NOT NULL,
      state TEXT NOT NULL,
      payload TEXT NOT NULL,
      created_at REAL NOT NULL,
      updated_at REAL NOT NULL
    );
    CREATE TABLE IF NOT EXISTS langgraph_store_items (
      namespace TEXT NOT NULL,
      key TEXT NOT NULL,
      value TEXT NOT NULL,
      index_text TEXT,
      created_at REAL NOT NULL,
      updated_at REAL NOT NULL,
      PRIMARY KEY(namespace, key)
    );
    CREATE INDEX IF NOT EXISTS langgraph_store_namespace
      ON langgraph_store_items(namespace, updated_at DESC);
    CREATE VIRTUAL TABLE IF NOT EXISTS langgraph_store_fts USING fts5(
      namespace UNINDEXED, key UNINDEXED, body, tokenize='unicode61'
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS provider_capability_probes (
      id TEXT PRIMARY KEY,
      configuration TEXT NOT NULL,
      capability TEXT NOT NULL,
      status TEXT NOT NULL,
      request_payload TEXT NOT NULL,
      response_payload TEXT,
      conclusion TEXT NOT NULL,
      started_at REAL NOT NULL,
      completed_at REAL NOT NULL
    );
    CREATE INDEX IF NOT EXISTS provider_capability_probes_time
      ON provider_capability_probes(completed_at DESC);
    """,
    """
    CREATE TABLE IF NOT EXISTS attachment_references (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      channel_id TEXT NOT NULL,
      message_id TEXT NOT NULL,
      attachment_id TEXT NOT NULL,
      filename TEXT NOT NULL,
      object_sha256 TEXT NOT NULL REFERENCES attachment_objects(sha256) ON DELETE CASCADE,
      metadata TEXT NOT NULL DEFAULT '{}',
      created_at REAL NOT NULL,
      UNIQUE(message_id, attachment_id)
    );
    CREATE INDEX IF NOT EXISTS attachment_references_chat
      ON attachment_references(channel_id, created_at DESC);
    """,
    """
    CREATE TABLE IF NOT EXISTS legacy_import_records (
      kind TEXT NOT NULL,
      source_id TEXT NOT NULL,
      payload TEXT NOT NULL,
      imported_at REAL NOT NULL,
      PRIMARY KEY(kind, source_id)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS config (
      key TEXT PRIMARY KEY, value TEXT NOT NULL, secret INTEGER NOT NULL DEFAULT 0,
      updated_at REAL NOT NULL
    );
    CREATE TABLE IF NOT EXISTS conversations (
      channel_id TEXT PRIMARY KEY, peer_id TEXT NOT NULL, peer_name TEXT NOT NULL,
      paused INTEGER NOT NULL DEFAULT 0, paused_at REAL, updated_at REAL NOT NULL,
      snoozed_until REAL, mode TEXT NOT NULL DEFAULT 'automatic'
    );
    CREATE TABLE IF NOT EXISTS messages (
      id TEXT PRIMARY KEY, channel_id TEXT NOT NULL, author_id TEXT NOT NULL,
      author_name TEXT NOT NULL, direction TEXT NOT NULL, source TEXT NOT NULL,
      content TEXT NOT NULL, timestamp REAL NOT NULL,
      attachments TEXT NOT NULL DEFAULT '[]'
    );
    CREATE INDEX IF NOT EXISTS messages_channel_time ON messages(channel_id, timestamp DESC);
    CREATE TABLE IF NOT EXISTS bot_nonces (nonce TEXT PRIMARY KEY, created_at REAL NOT NULL);
    CREATE TABLE IF NOT EXISTS bot_message_ids (id TEXT PRIMARY KEY, created_at REAL NOT NULL);
    CREATE TABLE IF NOT EXISTS assistant_reactions (
      trigger_message_id TEXT PRIMARY KEY, channel_id TEXT NOT NULL,
      emoji TEXT NOT NULL, created_at REAL NOT NULL
    );
    CREATE INDEX IF NOT EXISTS assistant_reactions_channel_time
      ON assistant_reactions(channel_id, created_at DESC);
    """,
    """
    UPDATE agent_configuration_versions
    SET configuration = json_set(
      json_remove(configuration, '$.options.max_completion_tokens'),
      '$.capabilities.output_token_limit',
      json('false')
    )
    WHERE json_extract(configuration, '$.provider_id') = 'chatgpt_subscription';

    UPDATE agent_configuration_versions
    SET configuration = json_set(
      configuration,
      '$.capabilities.output_token_limit',
      json('true')
    )
    WHERE json_extract(configuration, '$.provider_id') != 'chatgpt_subscription'
      AND json_type(configuration, '$.capabilities.output_token_limit') IS NULL;
    """,
    """
    ALTER TABLE messages ADD COLUMN edited_at REAL;
    ALTER TABLE messages ADD COLUMN deleted_at REAL;
    ALTER TABLE messages ADD COLUMN reply_to_message_id TEXT;

    CREATE TABLE chat_thread_generations (
      thread_id TEXT PRIMARY KEY,
      channel_id TEXT NOT NULL,
      account_id TEXT NOT NULL,
      generation INTEGER NOT NULL,
      configuration_version_id INTEGER REFERENCES agent_configuration_versions(id),
      created_at REAL NOT NULL,
      closed_at REAL,
      close_reason TEXT,
      summary TEXT,
      UNIQUE(channel_id, generation)
    );
    CREATE INDEX chat_thread_generations_chat
      ON chat_thread_generations(channel_id, generation DESC);
    INSERT INTO chat_thread_generations(
      thread_id, channel_id, account_id, generation, configuration_version_id, created_at
    )
    SELECT thread_id, channel_id, account_id, generation,
           (SELECT id FROM agent_configuration_versions WHERE active=1), updated_at
    FROM chat_threads;

    CREATE TABLE checkpoint_index (
      thread_id TEXT NOT NULL REFERENCES chat_thread_generations(thread_id) ON DELETE CASCADE,
      checkpoint_id TEXT NOT NULL,
      parent_checkpoint_id TEXT,
      run_id TEXT REFERENCES agent_runs(id) ON DELETE SET NULL,
      created_at REAL NOT NULL,
      step INTEGER,
      source TEXT,
      message_count INTEGER NOT NULL DEFAULT 0,
      PRIMARY KEY(thread_id, checkpoint_id)
    );
    CREATE INDEX checkpoint_index_run ON checkpoint_index(run_id, created_at DESC);

    ALTER TABLE agent_runs ADD COLUMN configuration_version_id INTEGER
      REFERENCES agent_configuration_versions(id);
    ALTER TABLE agent_runs ADD COLUMN trigger_kind TEXT;
    ALTER TABLE agent_runs ADD COLUMN trigger_message_id TEXT;
    ALTER TABLE agent_runs ADD COLUMN first_checkpoint_id TEXT;
    ALTER TABLE agent_runs ADD COLUMN final_checkpoint_id TEXT;
    ALTER TABLE agent_runs ADD COLUMN model_call_count INTEGER NOT NULL DEFAULT 0;
    ALTER TABLE agent_runs ADD COLUMN tool_call_count INTEGER NOT NULL DEFAULT 0;
    ALTER TABLE agent_runs ADD COLUMN delivery_count INTEGER NOT NULL DEFAULT 0;
    ALTER TABLE agent_runs ADD COLUMN input_tokens INTEGER;
    ALTER TABLE agent_runs ADD COLUMN output_tokens INTEGER;
    CREATE INDEX agent_runs_status_time ON agent_runs(status, started_at DESC);
    CREATE INDEX agent_runs_channel_time ON agent_runs(channel_id, started_at DESC);

    CREATE TABLE admin_jobs (
      id TEXT PRIMARY KEY,
      type TEXT NOT NULL,
      schema_version INTEGER NOT NULL,
      status TEXT NOT NULL CHECK(status IN (
        'queued','running','cancellation_requested','succeeded','failed','cancelled'
      )),
      idempotency_key TEXT,
      requested_at REAL NOT NULL,
      started_at REAL,
      completed_at REAL,
      cancellation_requested_at REAL,
      lease_owner TEXT,
      lease_expires_at REAL,
      attempt_count INTEGER NOT NULL DEFAULT 0,
      progress_stage TEXT,
      progress_current INTEGER,
      progress_total INTEGER,
      input_payload TEXT NOT NULL,
      target_kind TEXT,
      target_id TEXT,
      result_kind TEXT,
      result_id TEXT,
      error_code TEXT,
      error_summary TEXT
    );
    CREATE INDEX admin_jobs_status_time ON admin_jobs(status, requested_at DESC);
    CREATE INDEX admin_jobs_target ON admin_jobs(target_kind, target_id, requested_at DESC);
    CREATE INDEX admin_jobs_lease ON admin_jobs(status, lease_expires_at);
    CREATE UNIQUE INDEX admin_jobs_active_idempotency
      ON admin_jobs(idempotency_key)
      WHERE idempotency_key IS NOT NULL
        AND status IN ('queued','running','cancellation_requested');
    CREATE TABLE admin_job_events (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      job_id TEXT NOT NULL REFERENCES admin_jobs(id) ON DELETE CASCADE,
      sequence INTEGER NOT NULL,
      occurred_at REAL NOT NULL,
      kind TEXT NOT NULL,
      payload TEXT NOT NULL DEFAULT '{}',
      UNIQUE(job_id, sequence)
    );
    CREATE INDEX admin_job_events_job ON admin_job_events(job_id, sequence);
    CREATE TABLE provider_setup_drafts (
      id TEXT PRIMARY KEY,
      payload TEXT NOT NULL,
      secret INTEGER NOT NULL DEFAULT 1,
      fingerprint TEXT NOT NULL,
      created_at REAL NOT NULL,
      expires_at REAL NOT NULL
    );
    CREATE INDEX provider_setup_drafts_expiry ON provider_setup_drafts(expires_at);
    """,
    """
    CREATE TABLE admin_job_inputs (
      id TEXT PRIMARY KEY,
      payload TEXT NOT NULL,
      secret INTEGER NOT NULL DEFAULT 1,
      created_at REAL NOT NULL,
      expires_at REAL NOT NULL
    );
    CREATE INDEX admin_job_inputs_expiry ON admin_job_inputs(expires_at);
    """,
    """
    CREATE TABLE conversation_mailbox (
      id TEXT PRIMARY KEY,
      channel_id TEXT NOT NULL,
      sequence INTEGER NOT NULL,
      kind TEXT NOT NULL,
      available_at REAL NOT NULL,
      observed_at REAL NOT NULL,
      payload TEXT NOT NULL,
      state TEXT NOT NULL CHECK(state IN (
        'pending','claimed','completed','cancelled','failed'
      )),
      run_id TEXT,
      injection_batch INTEGER,
      claimed_at REAL,
      completed_at REAL,
      failure TEXT,
      UNIQUE(channel_id, sequence)
    );
    CREATE INDEX conversation_mailbox_ready
      ON conversation_mailbox(channel_id, state, available_at, sequence);
    CREATE INDEX conversation_mailbox_run
      ON conversation_mailbox(run_id, state, sequence);

    INSERT INTO conversation_mailbox(
      id, channel_id, sequence, kind, available_at, observed_at, payload,
      state, run_id, injection_batch, claimed_at, completed_at
    )
    SELECT e.id, e.channel_id, e.sequence, e.kind, e.observed_at, e.observed_at, e.payload,
      CASE COALESCE(q.disposition, 'completed')
        WHEN 'pending' THEN 'pending'
        WHEN 'claimed' THEN 'claimed'
        ELSE 'completed'
      END,
      q.logical_request_id, q.injection_batch, q.claimed_at,
      COALESCE(q.completed_at,
        CASE WHEN q.event_id IS NULL THEN e.observed_at ELSE NULL END)
    FROM discord_events e
    LEFT JOIN chat_event_queue q ON q.event_id=e.id;

    CREATE TABLE outbound_actions (
      id TEXT PRIMARY KEY,
      batch_id TEXT NOT NULL,
      ordinal INTEGER NOT NULL,
      thread_id TEXT NOT NULL,
      channel_id TEXT NOT NULL,
      run_id TEXT NOT NULL,
      source_kind TEXT NOT NULL,
      source_id TEXT NOT NULL,
      kind TEXT NOT NULL CHECK(kind IN ('discord_message','discord_reaction')),
      payload TEXT NOT NULL,
      state TEXT NOT NULL CHECK(state IN (
        'pending','dispatching','succeeded','failed','ambiguous'
      )),
      result TEXT,
      remote_id TEXT,
      error_code TEXT,
      lease_owner TEXT,
      lease_expires_at REAL,
      created_at REAL NOT NULL,
      completed_at REAL,
      UNIQUE(batch_id, ordinal)
    );
    CREATE INDEX outbound_actions_run ON outbound_actions(run_id, created_at, ordinal);
    CREATE INDEX outbound_actions_state ON outbound_actions(state, lease_expires_at);
    CREATE INDEX outbound_actions_source
      ON outbound_actions(thread_id, source_kind, source_id, ordinal);

    INSERT INTO outbound_actions(
      id, batch_id, ordinal, thread_id, channel_id, run_id,
      source_kind, source_id, kind, payload, state, result,
      error_code, created_at, completed_at
    )
    SELECT
      'legacy:' || legacy.run_id || ':' || legacy.tool_call_id,
      'legacy:' || legacy.run_id || ':' || legacy.tool_call_id,
      0,
      COALESCE((SELECT thread_id FROM agent_runs WHERE id=legacy.run_id),
               'legacy:' || legacy.run_id),
      COALESCE(json_extract(legacy.request, '$.channel_id'),
               (SELECT channel_id FROM agent_runs WHERE id=legacy.run_id), ''),
      legacy.run_id,
      'legacy_tool',
      legacy.tool_call_id,
      CASE legacy.action
        WHEN 'react_to_message' THEN 'discord_reaction'
        ELSE 'discord_message'
      END,
      legacy.request,
      CASE legacy.state
        WHEN 'claimed' THEN 'ambiguous'
        WHEN 'ambiguous' THEN 'ambiguous'
        ELSE CASE
          WHEN EXISTS(
            SELECT 1 FROM json_each(legacy.result)
            WHERE json_extract(value, '$.status') != 'accepted'
          ) THEN 'failed'
          ELSE 'succeeded'
        END
      END,
      legacy.result,
      CASE WHEN legacy.state='claimed' THEN 'legacy_incomplete_attempt' ELSE NULL END,
      legacy.claimed_at,
      legacy.completed_at
    FROM side_effect_deliveries AS legacy;

    CREATE TABLE conversation_waits (
      id TEXT PRIMARY KEY,
      thread_id TEXT NOT NULL,
      channel_id TEXT NOT NULL,
      run_id TEXT NOT NULL,
      trace_id TEXT NOT NULL,
      tool_call_id TEXT NOT NULL,
      wake_event_id TEXT NOT NULL REFERENCES conversation_mailbox(id) ON DELETE CASCADE,
      state TEXT NOT NULL CHECK(state IN (
        'arming','scheduled','resuming','completed','cancelled','failed'
      )),
      resume_at REAL NOT NULL,
      created_at REAL NOT NULL,
      updated_at REAL NOT NULL,
      failure TEXT
    );
    CREATE INDEX conversation_waits_due ON conversation_waits(state, resume_at);
    CREATE UNIQUE INDEX conversation_waits_active_channel
      ON conversation_waits(channel_id)
      WHERE state IN ('arming','scheduled','resuming');
    """,
    """
    ALTER TABLE conversation_waits
      ADD COLUMN payload TEXT NOT NULL DEFAULT '{}';
    """,
    """
    DROP TABLE IF EXISTS chat_event_queue;
    DROP TABLE IF EXISTS discord_events;
    DROP TABLE IF EXISTS side_effect_deliveries;
    """,
    """
    CREATE TABLE conversations_v12 (
      channel_id TEXT PRIMARY KEY,
      peer_id TEXT NOT NULL,
      peer_name TEXT NOT NULL,
      availability TEXT NOT NULL CHECK(availability IN ('active','paused')),
      paused_at REAL,
      snoozed_until REAL,
      updated_at REAL NOT NULL
    );
    INSERT INTO conversations_v12(
      channel_id, peer_id, peer_name, availability, paused_at, snoozed_until, updated_at
    )
    SELECT channel_id, peer_id, peer_name,
      CASE WHEN paused=1 THEN 'paused' ELSE 'active' END,
      paused_at, snoozed_until, updated_at
    FROM conversations;

    CREATE TABLE chat_interaction_policies (
      channel_id TEXT PRIMARY KEY REFERENCES conversations_v12(channel_id) ON DELETE CASCADE,
      preset TEXT NOT NULL CHECK(preset IN ('autonomous','shared','on_invocation','manual')),
      trigger_rules TEXT NOT NULL,
      trigger_participants TEXT NOT NULL,
      owner_handoff TEXT NOT NULL,
      conversation_role TEXT NOT NULL CHECK(conversation_role IN (
        'owner_delegate','shared_assistant','owner_copilot'
      )),
      identity_marker TEXT NOT NULL CHECK(identity_marker IN ('configurable','forced')),
      delivery TEXT NOT NULL CHECK(delivery IN ('immediate','owner_approval','dashboard_only')),
      active_turn_input TEXT NOT NULL,
      invocation_snooze_behavior TEXT NOT NULL CHECK(invocation_snooze_behavior IN ('bypass','respect')),
      invocation_turn_lifetime TEXT NOT NULL CHECK(invocation_turn_lifetime='strict'),
      policy_version INTEGER NOT NULL,
      updated_at REAL NOT NULL
    );
    INSERT INTO chat_interaction_policies(
      channel_id, preset, trigger_rules, trigger_participants, owner_handoff,
      conversation_role, identity_marker, delivery, active_turn_input,
      invocation_snooze_behavior, invocation_turn_lifetime, policy_version, updated_at
    )
    SELECT c.channel_id,
      CASE WHEN c.mode='inline' THEN 'shared' ELSE 'autonomous' END,
      '[{"kind":"every_message","aliases":[],"attention_locales":[],"additional_attention_words":[],"allow_bare_alias":true,"literal":"","typo_tolerance":{"enabled":true,"maximum_distance":1,"minimum_alias_graphemes":6}}]',
      CASE WHEN c.mode='inline' THEN '["owner","peer"]' ELSE '["peer"]' END,
      CASE WHEN c.mode='inline'
        THEN '{"availability_transition":"none","active_run_action":"keep_or_inject"}'
        ELSE '{"availability_transition":"snooze","active_run_action":"cancel"}'
      END,
      CASE WHEN c.mode='inline' THEN 'shared_assistant' ELSE 'owner_delegate' END,
      CASE WHEN c.mode='inline' THEN 'forced' ELSE 'configurable' END,
      'immediate',
      json_object(
        'timing', CASE WHEN COALESCE(t.live_steering, 1)=1
          THEN 'inject_at_safe_points' ELSE 'queue_for_next_turn' END,
        'participants', CASE WHEN c.mode='inline'
          THEN json('["owner","peer"]') ELSE json('["peer"]') END
      ),
      'bypass', 'strict', 1, c.updated_at
    FROM conversations AS c
    LEFT JOIN chat_threads AS t ON t.channel_id=c.channel_id;

    CREATE TABLE conversation_events (
      id TEXT PRIMARY KEY,
      channel_id TEXT NOT NULL,
      sequence INTEGER NOT NULL,
      kind TEXT NOT NULL CHECK(kind IN ('message','edit','delete')),
      payload TEXT NOT NULL,
      observed_at REAL NOT NULL,
      admission_decision TEXT NOT NULL DEFAULT '{}',
      context_state TEXT NOT NULL CHECK(context_state IN ('unapplied','claimed','applied')),
      run_id TEXT,
      injection_batch INTEGER,
      claimed_at REAL,
      applied_at REAL,
      failure TEXT,
      UNIQUE(channel_id, sequence)
    );
    CREATE INDEX conversation_events_context
      ON conversation_events(channel_id, context_state, sequence);
    CREATE INDEX conversation_events_run
      ON conversation_events(run_id, context_state, sequence);
    INSERT INTO conversation_events(
      id, channel_id, sequence, kind, payload, observed_at, admission_decision, context_state,
      run_id, injection_batch, claimed_at, applied_at, failure
    )
    SELECT id, channel_id, sequence, kind, payload, observed_at,
      '{"reason":"legacy_history"}',
      CASE
        WHEN state='claimed' THEN 'claimed'
        WHEN state='completed' AND run_id IS NOT NULL THEN 'applied'
        ELSE 'unapplied'
      END,
      CASE WHEN state='claimed' THEN run_id ELSE NULL END,
      CASE WHEN state='claimed' THEN injection_batch ELSE NULL END,
      CASE WHEN state='claimed' THEN claimed_at ELSE NULL END,
      CASE WHEN state='completed' AND run_id IS NOT NULL THEN completed_at ELSE NULL END,
      failure
    FROM conversation_mailbox
    WHERE kind IN ('message','edit','delete');

    CREATE TABLE agent_work (
      id TEXT PRIMARY KEY,
      channel_id TEXT NOT NULL,
      kind TEXT NOT NULL CHECK(kind IN ('turn','force','continuation')),
      source_event_id TEXT REFERENCES conversation_events(id) ON DELETE SET NULL,
      trigger_kind TEXT NOT NULL,
      trigger_participant TEXT,
      policy_version INTEGER NOT NULL,
      policy_snapshot TEXT NOT NULL,
      available_at REAL NOT NULL,
      state TEXT NOT NULL CHECK(state IN ('pending','claimed','completed','cancelled','failed')),
      run_id TEXT,
      captured_through_sequence INTEGER,
      decision TEXT NOT NULL DEFAULT '{}',
      created_at REAL NOT NULL,
      claimed_at REAL,
      completed_at REAL,
      failure TEXT
    );
    CREATE INDEX agent_work_ready ON agent_work(channel_id, state, available_at, created_at);
    CREATE INDEX agent_work_run ON agent_work(run_id, state, created_at);
    INSERT INTO agent_work(
      id, channel_id, kind, source_event_id, trigger_kind, trigger_participant,
      policy_version, policy_snapshot, available_at, state, run_id,
      captured_through_sequence, decision, created_at, claimed_at, completed_at, failure
    )
    SELECT
      CASE WHEN kind IN ('continuation_due','force_reply') THEN id ELSE 'work:' || id END,
      channel_id,
      CASE kind WHEN 'continuation_due' THEN 'continuation'
                WHEN 'force_reply' THEN 'force' ELSE 'turn' END,
      CASE WHEN kind IN ('message','edit','delete') THEN id ELSE NULL END,
      kind,
      json_extract(payload, '$.participant_role'),
      1, '{}', available_at,
      state, run_id,
      CASE WHEN state='claimed' THEN sequence ELSE NULL END,
      '{"reason":"migrated_mailbox_work"}', observed_at, claimed_at, completed_at, failure
    FROM conversation_mailbox
    WHERE kind IN ('continuation_due','force_reply') OR state IN ('pending','claimed','failed','cancelled');

    ALTER TABLE conversation_waits RENAME TO conversation_waits_v11;
    CREATE TABLE conversation_waits (
      id TEXT PRIMARY KEY,
      thread_id TEXT NOT NULL,
      channel_id TEXT NOT NULL,
      run_id TEXT NOT NULL,
      trace_id TEXT NOT NULL,
      tool_call_id TEXT NOT NULL,
      wake_work_id TEXT NOT NULL REFERENCES agent_work(id) ON DELETE CASCADE,
      state TEXT NOT NULL CHECK(state IN (
        'arming','scheduled','resuming','completed','cancelled','failed'
      )),
      resume_at REAL NOT NULL,
      created_at REAL NOT NULL,
      updated_at REAL NOT NULL,
      failure TEXT,
      payload TEXT NOT NULL DEFAULT '{}'
    );
    INSERT INTO conversation_waits(
      id, thread_id, channel_id, run_id, trace_id, tool_call_id, wake_work_id,
      state, resume_at, created_at, updated_at, failure, payload
    )
    SELECT id, thread_id, channel_id, run_id, trace_id, tool_call_id, wake_event_id,
      state, resume_at, created_at, updated_at, failure, payload
    FROM conversation_waits_v11;
    DROP TABLE conversation_waits_v11;
    CREATE INDEX conversation_waits_due ON conversation_waits(state, resume_at);
    CREATE UNIQUE INDEX conversation_waits_active_channel
      ON conversation_waits(channel_id)
      WHERE state IN ('arming','scheduled','resuming');

    CREATE TABLE chat_threads_v12 (
      channel_id TEXT PRIMARY KEY,
      account_id TEXT NOT NULL,
      generation INTEGER NOT NULL DEFAULT 1,
      thread_id TEXT NOT NULL UNIQUE,
      applied_event_sequence INTEGER NOT NULL DEFAULT 0,
      updated_at REAL NOT NULL
    );
    INSERT INTO chat_threads_v12(
      channel_id, account_id, generation, thread_id, applied_event_sequence, updated_at
    )
    SELECT t.channel_id, t.account_id, t.generation, t.thread_id,
      COALESCE((
        SELECT MAX(e.sequence) FROM conversation_events AS e
        WHERE e.channel_id=t.channel_id AND e.context_state='applied'
          AND NOT EXISTS(
            SELECT 1 FROM conversation_events AS earlier
            WHERE earlier.channel_id=e.channel_id AND earlier.sequence<=e.sequence
              AND earlier.context_state!='applied'
          )
      ), 0),
      t.updated_at
    FROM chat_threads AS t;

    UPDATE escalation_interrupts
    SET payload=json_set(payload, '$.resume_strategy', 'journal')
    WHERE json_extract(payload, '$.resume_strategy')='mailbox';

    DROP TABLE conversation_mailbox;
    DROP TABLE chat_threads;
    ALTER TABLE chat_threads_v12 RENAME TO chat_threads;
    ALTER TABLE conversations RENAME TO conversations_v11;
    ALTER TABLE conversations_v12 RENAME TO conversations;
    DROP TABLE conversations_v11;
    """,
    """
    CREATE TABLE conversation_events_v13 (
      id TEXT PRIMARY KEY,
      channel_id TEXT NOT NULL,
      sequence INTEGER NOT NULL,
      kind TEXT NOT NULL CHECK(kind IN ('message','edit','delete','reaction')),
      payload TEXT NOT NULL,
      observed_at REAL NOT NULL,
      admission_decision TEXT NOT NULL DEFAULT '{}',
      context_state TEXT NOT NULL CHECK(context_state IN ('unapplied','claimed','applied')),
      run_id TEXT,
      injection_batch INTEGER,
      claimed_at REAL,
      applied_at REAL,
      failure TEXT,
      UNIQUE(channel_id, sequence)
    );
    INSERT INTO conversation_events_v13 SELECT * FROM conversation_events;

    CREATE TABLE agent_work_v13 (
      id TEXT PRIMARY KEY,
      channel_id TEXT NOT NULL,
      kind TEXT NOT NULL CHECK(kind IN ('turn','force','continuation')),
      source_event_id TEXT REFERENCES conversation_events_v13(id) ON DELETE SET NULL,
      trigger_kind TEXT NOT NULL,
      trigger_participant TEXT,
      policy_version INTEGER NOT NULL,
      policy_snapshot TEXT NOT NULL,
      available_at REAL NOT NULL,
      state TEXT NOT NULL CHECK(state IN ('pending','claimed','completed','cancelled','failed')),
      run_id TEXT,
      captured_through_sequence INTEGER,
      decision TEXT NOT NULL DEFAULT '{}',
      created_at REAL NOT NULL,
      claimed_at REAL,
      completed_at REAL,
      failure TEXT
    );
    INSERT INTO agent_work_v13 SELECT * FROM agent_work;

    CREATE TABLE conversation_waits_v13 (
      id TEXT PRIMARY KEY,
      thread_id TEXT NOT NULL,
      channel_id TEXT NOT NULL,
      run_id TEXT NOT NULL,
      trace_id TEXT NOT NULL,
      tool_call_id TEXT NOT NULL,
      wake_work_id TEXT NOT NULL REFERENCES agent_work_v13(id) ON DELETE CASCADE,
      state TEXT NOT NULL CHECK(state IN (
        'arming','scheduled','resuming','completed','cancelled','failed'
      )),
      resume_at REAL NOT NULL,
      created_at REAL NOT NULL,
      updated_at REAL NOT NULL,
      failure TEXT,
      payload TEXT NOT NULL DEFAULT '{}'
    );
    INSERT INTO conversation_waits_v13 SELECT * FROM conversation_waits;

    DROP TABLE conversation_waits;
    DROP TABLE agent_work;
    DROP TABLE conversation_events;
    ALTER TABLE conversation_events_v13 RENAME TO conversation_events;
    ALTER TABLE agent_work_v13 RENAME TO agent_work;
    ALTER TABLE conversation_waits_v13 RENAME TO conversation_waits;
    CREATE INDEX conversation_events_context
      ON conversation_events(channel_id, context_state, sequence);
    CREATE INDEX conversation_events_run
      ON conversation_events(run_id, context_state, sequence);
    CREATE INDEX agent_work_ready ON agent_work(channel_id, state, available_at, created_at);
    CREATE INDEX agent_work_run ON agent_work(run_id, state, created_at);
    CREATE INDEX conversation_waits_due ON conversation_waits(state, resume_at);
    CREATE UNIQUE INDEX conversation_waits_active_channel
      ON conversation_waits(channel_id)
      WHERE state IN ('arming','scheduled','resuming');
    """,
    """
    ALTER TABLE chat_interaction_policies
      ADD COLUMN availability_schedule TEXT NOT NULL
      DEFAULT '{"enabled":false,"weekdays":[0,1,2,3,4,5,6],"start_minute":540,"end_minute":1020,"timezone":""}';
    """,
)


class AsyncSQLite:
    """One serialized async connection for Diskovod's application repositories."""

    def __init__(self, path: Path):
        self.path = path
        self._connection: aiosqlite.Connection | None = None
        self._connection_lock = asyncio.Lock()
        self._transaction_lock = asyncio.Lock()

    async def start(self) -> None:
        await self._get_connection()

    async def close(self) -> None:
        async with self._transaction_lock:
            async with self._connection_lock:
                connection, self._connection = self._connection, None
            if connection is not None:
                await connection.close()

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[aiosqlite.Connection]:
        """Serialize a complete application transaction on the shared connection."""
        async with self._transaction_lock:
            connection = await self._get_connection()
            try:
                yield connection
            except BaseException:
                await connection.rollback()
                raise
            else:
                await connection.commit()

    async def _get_connection(self) -> aiosqlite.Connection:
        if self._connection is not None:
            return self._connection
        async with self._connection_lock:
            if self._connection is None:
                connection = await aiosqlite.connect(self.path)
                connection.row_factory = aiosqlite.Row
                await connection.execute(f"PRAGMA busy_timeout={SQLITE_BUSY_TIMEOUT_MS}")
                await connection.execute("PRAGMA foreign_keys=ON")
                self._connection = connection
        return self._connection


async def initialize_target_schema(connection: aiosqlite.Connection) -> None:
    """Apply target-schema migrations to Diskovod's single relational database."""
    await connection.execute("PRAGMA journal_mode=WAL")
    await connection.execute(f"PRAGMA busy_timeout={SQLITE_BUSY_TIMEOUT_MS}")
    await connection.execute("PRAGMA foreign_keys=ON")
    await connection.execute(
        "CREATE TABLE IF NOT EXISTS schema_migrations (version INTEGER PRIMARY KEY, applied_at REAL NOT NULL)"
    )
    applied = {
        int(row[0])
        for row in await (await connection.execute("SELECT version FROM schema_migrations")).fetchall()
    }
    for version, migration in enumerate(TARGET_MIGRATIONS, start=1):
        if version in applied:
            continue
        await connection.executescript(migration)
        await connection.execute(
            "INSERT INTO schema_migrations(version, applied_at) VALUES(?, ?)",
            (version, datetime.now(UTC).timestamp()),
        )
    current = (
        await (await connection.execute("SELECT COALESCE(MAX(version), 0) FROM schema_migrations")).fetchone()
    )[0]
    if current != TARGET_SCHEMA_VERSION:
        raise RuntimeError(f"Unsupported Diskovod schema version {current}")


class CheckpointCipher:
    """AES-GCM checkpoint cipher derived from Diskovod's existing secret."""

    name = "diskovod-aesgcm-v1"

    def __init__(self, secret: str):
        if len(secret) < 32:
            raise ValueError("The secret key file must contain at least 32 characters")
        self._cipher = AESGCM(hashlib.sha256(b"diskovod-checkpoints\0" + secret.encode()).digest())

    def encrypt(self, plaintext: bytes) -> tuple[str, bytes]:
        nonce = os.urandom(12)
        ciphertext = self._cipher.encrypt(nonce, plaintext, self.name.encode())
        return self.name, nonce + ciphertext

    def decrypt(self, ciphername: str, ciphertext: bytes) -> bytes:
        if ciphername != self.name:
            raise ValueError(f"Unsupported checkpoint cipher {ciphername!r}")
        if len(ciphertext) < 13:
            raise ValueError("Invalid encrypted checkpoint")
        return self._cipher.decrypt(ciphertext[:12], ciphertext[12:], self.name.encode())


def checkpoint_serializer(secret: str) -> EncryptedSerializer:
    serde = JsonPlusSerializer(pickle_fallback=False, allowed_msgpack_modules=None)
    return EncryptedSerializer(CheckpointCipher(secret), serde=serde)


@asynccontextmanager
async def open_checkpointer(path: Path, secret: str) -> AsyncIterator[AsyncSqliteSaver]:
    """Open an encrypted LangGraph checkpointer on the shared Diskovod database."""
    async with aiosqlite.connect(path) as connection:
        await connection.execute(f"PRAGMA busy_timeout={SQLITE_BUSY_TIMEOUT_MS}")
        await connection.execute("PRAGMA foreign_keys=ON")
        saver = AsyncSqliteSaver(connection, serde=checkpoint_serializer(secret))
        await saver.setup()
        yield saver


class SQLiteLangGraphStore(BaseStore):
    """Persistent local LangGraph Store with JSON filtering and lexical FTS search."""

    supports_ttl = False

    def __init__(self, path: Path, database: AsyncSQLite | None = None):
        self.path = path
        self.database = database or AsyncSQLite(path)
        self._schema_ready = False
        self._schema_lock = asyncio.Lock()

    def batch(self, ops: Iterable[Op]) -> list[Result]:
        del ops
        raise NotImplementedError("Diskovod's LangGraph Store supports only the asynchronous API")

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        await self.start()
        return await self._abatch(self.database, list(ops))

    async def start(self) -> None:
        if self._schema_ready:
            return
        async with self._schema_lock:
            if not self._schema_ready:
                await self._initialize_database(self.database)
                self._schema_ready = True

    @staticmethod
    async def _initialize_database(database: AsyncSQLite) -> None:
        async with database.transaction() as connection:
            await initialize_target_schema(connection)

    async def _abatch(self, database: AsyncSQLite, operations: list[Op]) -> list[Result]:
        results: list[Result] = []
        async with database.transaction() as connection:
            for operation in operations:
                if isinstance(operation, GetOp):
                    results.append(await self._aget(connection, operation))
                elif isinstance(operation, PutOp):
                    results.append(await self._aput(connection, operation))
                elif isinstance(operation, SearchOp):
                    results.append(await self._asearch(connection, operation))
                elif isinstance(operation, ListNamespacesOp):
                    results.append(await self._alist_namespaces(connection, operation))
                else:
                    raise TypeError(f"Unsupported Store operation: {type(operation).__name__}")
        return results

    async def _aget(self, connection: aiosqlite.Connection, operation: GetOp) -> Item | None:
        namespace = self._namespace(operation.namespace)
        row = await (
            await connection.execute(
                "SELECT * FROM langgraph_store_items WHERE namespace=? AND key=?",
                (namespace, operation.key),
            )
        ).fetchone()
        return self._item(row) if row else None

    async def _aput(self, connection: aiosqlite.Connection, operation: PutOp) -> None:
        namespace = self._namespace(operation.namespace)
        if not operation.key:
            raise ValueError("Store keys cannot be empty")
        await connection.execute(
            "DELETE FROM langgraph_store_fts WHERE namespace=? AND key=?",
            (namespace, operation.key),
        )
        if operation.value is None:
            await connection.execute(
                "DELETE FROM langgraph_store_items WHERE namespace=? AND key=?",
                (namespace, operation.key),
            )
            return None
        value = self._json(operation.value)
        index_text = self._index_text(operation.value, operation.index)
        now = datetime.now(UTC).timestamp()
        await connection.execute(
            """
            INSERT INTO langgraph_store_items(namespace, key, value, index_text, created_at, updated_at)
            VALUES(?, ?, ?, ?, ?, ?)
            ON CONFLICT(namespace, key) DO UPDATE SET
              value=excluded.value, index_text=excluded.index_text, updated_at=excluded.updated_at
            """,
            (namespace, operation.key, value, index_text, now, now),
        )
        if index_text:
            await connection.execute(
                "INSERT INTO langgraph_store_fts(namespace, key, body) VALUES(?, ?, ?)",
                (namespace, operation.key, index_text),
            )
        return None

    async def _asearch(self, connection: aiosqlite.Connection, operation: SearchOp) -> list[SearchItem]:
        rows = await (
            await connection.execute(
                "SELECT * FROM langgraph_store_items ORDER BY updated_at DESC, namespace, key"
            )
        ).fetchall()
        prefix = operation.namespace_prefix
        candidates = [
            row for row in rows if self._decode_namespace(row["namespace"])[: len(prefix)] == prefix
        ]
        if operation.filter:
            candidates = [
                row for row in candidates if self._matches_filter(json.loads(row["value"]), operation.filter)
            ]
        scores: dict[tuple[str, str], float] = {}
        if operation.query:
            query = self._fts_query(operation.query)
            if not query:
                return []
            matches = await (
                await connection.execute(
                    "SELECT namespace, key, bm25(langgraph_store_fts) AS rank "
                    "FROM langgraph_store_fts WHERE body MATCH ?",
                    (query,),
                )
            ).fetchall()
            scores = {(row["namespace"], row["key"]): -float(row["rank"]) for row in matches}
            candidates = [row for row in candidates if (row["namespace"], row["key"]) in scores]
            candidates.sort(
                key=lambda row: (scores[(row["namespace"], row["key"])], row["updated_at"]),
                reverse=True,
            )
        selected = candidates[operation.offset : operation.offset + operation.limit]
        return [
            SearchItem(
                namespace=self._decode_namespace(row["namespace"]),
                key=row["key"],
                value=json.loads(row["value"]),
                created_at=self._datetime(row["created_at"]),
                updated_at=self._datetime(row["updated_at"]),
                score=scores.get((row["namespace"], row["key"])),
            )
            for row in selected
        ]

    async def _alist_namespaces(
        self, connection: aiosqlite.Connection, operation: ListNamespacesOp
    ) -> list[tuple[str, ...]]:
        rows = await (
            await connection.execute("SELECT DISTINCT namespace FROM langgraph_store_items")
        ).fetchall()
        namespaces = [self._decode_namespace(row["namespace"]) for row in rows]
        if operation.match_conditions:
            namespaces = [
                namespace
                for namespace in namespaces
                if all(
                    self._matches_namespace(namespace, condition.match_type, condition.path)
                    for condition in operation.match_conditions
                )
            ]
        if operation.max_depth is not None:
            namespaces = list({namespace[: operation.max_depth] for namespace in namespaces})
        namespaces.sort()
        return namespaces[operation.offset : operation.offset + operation.limit]

    @classmethod
    def _item(cls, row: aiosqlite.Row) -> Item:
        return Item(
            namespace=cls._decode_namespace(row["namespace"]),
            key=row["key"],
            value=json.loads(row["value"]),
            created_at=cls._datetime(row["created_at"]),
            updated_at=cls._datetime(row["updated_at"]),
        )

    @staticmethod
    def _namespace(namespace: tuple[str, ...]) -> str:
        if not namespace:
            raise InvalidNamespaceError("Namespace cannot be empty")
        if namespace[0] == "langgraph":
            raise InvalidNamespaceError('Root namespace label cannot be "langgraph"')
        if any(not isinstance(label, str) or not label or "." in label for label in namespace):
            raise InvalidNamespaceError("Namespace labels must be non-empty strings without periods")
        return json.dumps(namespace, ensure_ascii=False, separators=(",", ":"))

    @staticmethod
    def _decode_namespace(value: str) -> tuple[str, ...]:
        return tuple(json.loads(value))

    @staticmethod
    def _json(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)

    @classmethod
    def _index_text(cls, value: dict[str, Any], index: bool | list[str] | None) -> str | None:
        if index is False:
            return None
        selected: Any = value
        if isinstance(index, list):
            selected = [cls._path_value(value, path) for path in index]
        parts: list[str] = []

        def visit(item: Any) -> None:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                for child in item.values():
                    visit(child)
            elif isinstance(item, (list, tuple)):
                for child in item:
                    visit(child)
            elif item is not None:
                parts.append(str(item))

        visit(selected)
        return "\n".join(parts) or None

    @staticmethod
    def _path_value(value: dict[str, Any], path: str) -> Any:
        current: Any = value
        normalized = path[2:] if path.startswith("$.") else path
        for component in normalized.split("."):
            if not isinstance(current, dict):
                return None
            current = current.get(component)
        return current

    @classmethod
    def _matches_filter(cls, value: Any, expected: Any) -> bool:
        if isinstance(expected, dict):
            if any(str(key).startswith("$") for key in expected):
                return all(
                    cls._matches_operator(value, operator, operand) for operator, operand in expected.items()
                )
            if not isinstance(value, dict):
                return False
            return all(
                key in value and cls._matches_filter(value[key], child) for key, child in expected.items()
            )
        if isinstance(expected, (list, tuple)):
            return (
                isinstance(value, (list, tuple))
                and len(value) == len(expected)
                and all(cls._matches_filter(item, child) for item, child in zip(value, expected, strict=True))
            )
        return value == expected

    @staticmethod
    def _matches_operator(value: Any, operator: str, operand: Any) -> bool:
        if operator == "$eq":
            return value == operand
        if operator == "$ne":
            return value != operand
        if operator in {"$gt", "$gte", "$lt", "$lte"}:
            left = float(value)
            right = float(operand)
            return {
                "$gt": left > right,
                "$gte": left >= right,
                "$lt": left < right,
                "$lte": left <= right,
            }[operator]
        raise ValueError(f"Unsupported filter operator: {operator}")

    @staticmethod
    def _matches_namespace(namespace: tuple[str, ...], match_type: str, path: tuple[str, ...]) -> bool:
        if len(namespace) < len(path):
            return False
        candidate = namespace[: len(path)] if match_type == "prefix" else namespace[-len(path) :]
        return all(pattern == "*" or pattern == value for value, pattern in zip(candidate, path, strict=True))

    @staticmethod
    def _fts_query(query: str) -> str:
        tokens = [token.replace('"', '""') for token in query.split() if token]
        return " AND ".join(f'"{token}"' for token in tokens)

    @staticmethod
    def _datetime(timestamp: float) -> datetime:
        return datetime.fromtimestamp(timestamp, UTC)
