from __future__ import annotations

from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field

from .admin_jobs import AdminJobContext, AdminJobService, JobDefinition, JobResult
from .discord import DiscordService
from .localization import prompts_for
from .personality import assistant_profile_fingerprint, personality_source_hash
from .providers import ModelConfiguration, ModelService, ProviderCredentials, ProviderSetup
from .runtime import AgentService
from .store import Store


class CapabilityProbePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    configuration_id: int = Field(gt=0)
    capability: Literal["native_tools", "hosted_web_search"]
    apply_result: bool = True


class SetupDraftProbePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    draft_id: str = Field(min_length=1, max_length=200)
    capability: Literal["native_tools", "hosted_web_search"]


class PersonalityInferencePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    configuration_id: int = Field(gt=0)
    prompt_locale: str = Field(min_length=2, max_length=10)
    profile_fingerprint: str = Field(min_length=64, max_length=64)
    source: Literal["pasted_history", "discord_history"]
    history_limit: int = Field(default=100, ge=20, le=500)
    input_id: str | None = Field(default=None, min_length=1, max_length=200)


class CheckpointReplayPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    thread_id: str = Field(min_length=1, max_length=500)
    checkpoint_id: str = Field(min_length=1, max_length=500)
    configuration_id: int = Field(gt=0)


def register_provider_jobs(
    jobs: AdminJobService,
    store: Store,
    models: ModelService,
    provider_setup: ProviderSetup,
) -> None:
    async def probe_active(context: AdminJobContext, raw_payload: BaseModel) -> JobResult | None:
        payload = CapabilityProbePayload.model_validate(raw_payload)
        configuration = await store.aagent_configuration(payload.configuration_id)
        if configuration is None:
            raise ValueError("The model configuration no longer exists")
        await context.progress("building_probe_request")
        credentials = models.credentials_for(configuration)
        await context.progress(f"testing_{payload.capability}")
        if payload.capability == "native_tools":
            probe = await provider_setup.probe_client_tools(configuration, credentials)
        else:
            probe = await provider_setup.probe_hosted_web_search(configuration, credentials)
        await context.checkpoint()
        if payload.apply_result and await store.aactive_configuration_id() == payload.configuration_id:
            await store.asave_agent_configuration(
                provider_setup.configuration_with_capability(
                    configuration,
                    payload.capability,
                    probe.supported,
                    probe.id,
                )
            )
        return JobResult("provider_capability_probe", probe.id)

    async def probe_draft(context: AdminJobContext, raw_payload: BaseModel) -> JobResult | None:
        payload = SetupDraftProbePayload.model_validate(raw_payload)
        draft = await store.aprovider_setup_draft(payload.draft_id)
        if draft is None:
            raise ValueError("The provider setup draft expired or does not exist")
        values = draft["payload"]
        configuration = ModelConfiguration.from_dict(values["configuration"])
        credentials = ProviderCredentials(**values["credentials"])
        await context.progress(f"testing_{payload.capability}")
        if payload.capability == "native_tools":
            probe = await provider_setup.probe_client_tools(configuration, credentials)
        else:
            probe = await provider_setup.probe_hosted_web_search(configuration, credentials)
        await context.checkpoint()
        capabilities = dict(values.get("capabilities") or {})
        capabilities[payload.capability] = probe.supported
        values["capabilities"] = capabilities
        values["last_probe_id"] = probe.id
        if not await store.aupdate_provider_setup_draft(payload.draft_id, values):
            raise ValueError("The provider setup draft expired before the result could be saved")
        return JobResult("provider_capability_probe", probe.id)

    jobs.register(
        "provider.capability_probe",
        JobDefinition(CapabilityProbePayload, probe_active, retryable=True, cancellable=True),
    )
    jobs.register(
        "provider.setup_draft_probe",
        JobDefinition(SetupDraftProbePayload, probe_draft, retryable=True, cancellable=True),
    )


def register_runtime_jobs(
    jobs: AdminJobService,
    store: Store,
    models: ModelService,
    discord: DiscordService,
    runtime: AgentService,
) -> None:
    async def infer_personality(context: AdminJobContext, raw_payload: BaseModel) -> JobResult | None:
        payload = PersonalityInferencePayload.model_validate(raw_payload)
        await context.progress("loading_personality_samples")
        if payload.source == "discord_history":
            messages = await discord.personality_history(payload.history_limit)
            samples = "\n\n---\n\n".join(messages)
        else:
            stored = await store.aadmin_job_input(payload.input_id or "")
            if stored is None:
                raise ValueError("The encrypted personality input expired or does not exist")
            samples = str(stored.get("samples") or "")
        if len(samples) < 200:
            raise ValueError("Not enough representative message history was available")

        configuration = await store.aagent_configuration(payload.configuration_id)
        if configuration is None:
            raise ValueError("The model configuration no longer exists")
        await context.progress("inferring_personality")
        response = await models.build_configuration(
            configuration,
            models.credentials_for(configuration),
        ).ainvoke([SystemMessage(prompts_for(payload.prompt_locale).personality), HumanMessage(samples)])
        profile = response.text.strip()
        await context.checkpoint()
        current = store.assistant_profile()
        if assistant_profile_fingerprint(current) != payload.profile_fingerprint:
            raise ValueError("Assistant settings changed while personality inference was running")
        source_hash = personality_source_hash(samples, payload.prompt_locale)
        await store.aset_personality(profile, source_hash, source=payload.source)
        await models.arefresh_prompt_cache_identity()
        if payload.input_id:
            await store.adelete_admin_job_input(payload.input_id)
        return JobResult("assistant_personality", source_hash)

    async def replay_checkpoint(context: AdminJobContext, raw_payload: BaseModel) -> JobResult | None:
        payload = CheckpointReplayPayload.model_validate(raw_payload)
        await context.progress("loading_checkpoint")
        await context.checkpoint()
        await context.progress("replaying_checkpoint")
        run_id = await runtime.replay_checkpoint(
            payload.thread_id,
            payload.checkpoint_id,
            configuration_id=payload.configuration_id,
        )
        return JobResult("agent_run", run_id)

    jobs.register(
        "assistant.personality_inference",
        JobDefinition(PersonalityInferencePayload, infer_personality, retryable=True, cancellable=True),
    )
    jobs.register(
        "runtime.checkpoint_replay",
        JobDefinition(CheckpointReplayPayload, replay_checkpoint, retryable=False, cancellable=True),
    )
