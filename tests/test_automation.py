import asyncio
import time
from pathlib import Path
from typing import cast

import pytest

from diskovod.automation import Automation
from diskovod.chatgpt import ChatGPTClient
from diskovod.store import Store


@pytest.mark.asyncio
async def test_human_activity_cancels_inflight_work_and_snoozes(tmp_path: Path):
    store = Store(tmp_path / "state.sqlite3", "x" * 32)
    store.upsert_conversation("dm", "peer", "Peer")
    automation = Automation(store, cast(ChatGPTClient, None))
    task = asyncio.create_task(asyncio.sleep(60))
    automation.tasks["dm"] = task

    snoozed_until = automation.human_activity("dm")
    await asyncio.gather(task, return_exceptions=True)

    assert task.cancelled()
    conversation = store.conversation("dm")
    assert conversation["paused"] is False
    assert conversation["snoozed_until"] is not None
    assert store.can_automate("dm") is False
    assert 15 * 60 - 1 <= snoozed_until - time.time() <= 30 * 60 + 1
    assert automation.versions["dm"] == 1
    store.close()


def test_permanent_pause_is_separate_from_human_quiet_window(tmp_path: Path):
    store = Store(tmp_path / "state.sqlite3", "x" * 32)
    store.upsert_conversation("dm", "peer", "Peer")
    automation = Automation(store, cast(ChatGPTClient, None))

    automation.permanently_pause("dm")

    assert store.conversation("dm")["paused"] is True
    assert store.conversation("dm")["snoozed_until"] is None
    store.close()
