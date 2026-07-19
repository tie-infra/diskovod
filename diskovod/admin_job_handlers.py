from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .admin_jobs import AdminJobContext, AdminJobService, JobDefinition, JobResult
from .providers import ModelConfiguration, ModelService, ProviderCredentials, ProviderSetup
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
