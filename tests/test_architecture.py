from __future__ import annotations

import ast
from pathlib import Path

import pytest


PACKAGE_ROOT = Path(__file__).parents[1] / "diskovod"


def _class_node(module: str, name: str) -> ast.ClassDef:
    tree = ast.parse((PACKAGE_ROOT / module).read_text())
    return next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)


def _members(module: str, name: str) -> set[str]:
    class_node = _class_node(module, name)
    members = {
        node.name
        for node in class_node.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    for node in ast.walk(class_node):
        if isinstance(node, ast.AnnAssign):
            targets = (node.target,)
        elif isinstance(node, ast.Assign):
            targets = tuple(node.targets)
        else:
            continue
        for target in targets:
            if (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
            ):
                members.add(target.attr)
    return members


def _service_references(module: str, owner: str, attribute: str) -> dict[str, int]:
    references: dict[str, int] = {}
    for node in ast.walk(_class_node(module, owner)):
        if not isinstance(node, ast.Attribute) or not isinstance(node.value, ast.Attribute):
            continue
        receiver = node.value
        if (
            isinstance(receiver.value, ast.Name)
            and receiver.value.id == "self"
            and receiver.attr == attribute
        ):
            references.setdefault(node.attr, node.lineno)
    return references


@pytest.mark.parametrize(
    ("owner_module", "owner", "attribute", "target_module", "target"),
    [
        ("discord.py", "PrivateDiscordClient", "runtime", "runtime.py", "AgentService"),
        ("discord.py", "PrivateDiscordClient", "store", "store.py", "Store"),
        ("discord.py", "DiscordService", "runtime", "runtime.py", "AgentService"),
        ("discord.py", "DiscordService", "store", "store.py", "Store"),
        ("web.py", "WebApp", "runtime", "runtime.py", "AgentService"),
        ("web.py", "WebApp", "store", "store.py", "Store"),
        ("web.py", "WebApp", "models", "providers/service.py", "ModelService"),
        ("web.py", "WebApp", "provider_setup", "providers/setup.py", "ProviderSetup"),
        ("web.py", "WebApp", "discord", "discord.py", "DiscordService"),
        ("web.py", "WebApp", "queries", "admin_queries.py", "AdminQueryService"),
        ("web.py", "WebApp", "jobs", "admin_jobs.py", "AdminJobService"),
        ("migration.py", "LegacyMigrator", "runtime", "runtime.py", "AgentService"),
        ("migration.py", "LegacyMigrator", "store", "store.py", "Store"),
        ("runtime.py", "AgentService", "store", "store.py", "Store"),
        ("runtime.py", "AgentService", "models", "providers/service.py", "ModelService"),
        ("runtime.py", "AgentService", "mailbox", "mailbox.py", "ConversationMailbox"),
        ("runtime.py", "AgentService", "attachments", "attachments.py", "AttachmentRepository"),
        ("runtime.py", "AgentService", "publisher", "outbound.py", "OutboundPublisher"),
        ("runtime.py", "AgentService", "waits", "waits.py", "ConversationWaits"),
        ("runtime.py", "AgentService", "memory", "persistence.py", "SQLiteLangGraphStore"),
    ],
)
def test_internal_service_paths_resolve(
    owner_module: str,
    owner: str,
    attribute: str,
    target_module: str,
    target: str,
):
    available = _members(target_module, target)
    references = _service_references(owner_module, owner, attribute)

    missing = {
        member: line
        for member, line in references.items()
        if member not in available
    }
    assert missing == {}, (
        f"{owner_module}:{owner} has invalid self.{attribute} references for "
        f"{target_module}:{target}: {missing}"
    )
