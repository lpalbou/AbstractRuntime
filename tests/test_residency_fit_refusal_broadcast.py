"""A load the fit check refuses (core ADR 0010 `ModelDoesNotFitError`): never
retried (a retry would only evict more), the sentence is the effect error, and
its numbers and suggestions reach the ledger and the residency diagnostics."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

import abstractruntime.integrations.abstractcore.llm_client as llm_mod
from abstractruntime.core.models import StepRecord, StepStatus
from abstractruntime.core.runtime import EffectOutcome
from abstractruntime.integrations.abstractcore.llm_client import MultiLocalAbstractCoreLLMClient


class _DoesNotFit(Exception):
    code = "model_does_not_fit"

    def as_dict(self):
        return {"code": self.code, "message": str(self), "needed_bytes": 13_700_000_000,
                "free_bytes": 3_000_000_000, "suggestions": [{"artifact": "org/m:Q4_K_M", "download_bytes": 5_600_000_000}]}


SENTENCE = "org/m needs 13.7 GB but only 3.0 GB is free on the GPU. Choose a smaller model, for example org/m:Q4_K_M (5.6 GB)."


def test_refusal_details_found_through_the_cause_chain():
    try:
        try:
            raise _DoesNotFit(SENTENCE)
        except _DoesNotFit as inner:
            raise RuntimeError("wrapped") from inner
    except RuntimeError as outer:
        d = llm_mod.residency_refusal_details(outer)
    assert d["code"] == "model_does_not_fit" and d["needed_bytes"] == 13_700_000_000
    assert llm_mod.residency_refusal_details(ValueError("x")) is None


def test_the_pool_keeps_the_refusal_for_the_models_page(monkeypatch):
    def refuse(self, provider, model, *, llm_kwargs_override=None):
        raise _DoesNotFit(SENTENCE)

    monkeypatch.setattr(MultiLocalAbstractCoreLLMClient, "_create_client", refuse)
    monkeypatch.setattr(llm_mod, "_WARM_DEFAULT_MODEL", False)
    c = MultiLocalAbstractCoreLLMClient(provider="huggingface", model="org/default")
    with pytest.raises(_DoesNotFit):
        c._get_client("huggingface", "org/m")
    refusals = c.residency_eject_diagnostics()["refusals"]
    assert refusals and refusals[-1]["message"] == SENTENCE, "MUTANT: refusal not kept -> RED"
    with pytest.raises(_DoesNotFit) as err:      # the default pair: the sentence, not a config hint
        c._get_client("huggingface", "org/default")
    assert str(err.value) == SENTENCE


def test_step_record_keeps_the_typed_fields():
    rec = StepRecord(run_id="r", step_id="s", node_id="n", status=StepStatus.STARTED) if hasattr(StepStatus, "STARTED") else None
    if rec is None:
        pytest.skip("StepStatus has no STARTED")
    rec.finish_failure(SENTENCE, details={"code": "model_does_not_fit", "needed_bytes": 1})
    assert rec.error == SENTENCE and rec.result == {"error_details": {"code": "model_does_not_fit", "needed_bytes": 1}}
    out = EffectOutcome.failed(SENTENCE, retryable=False, error_details={"code": "model_does_not_fit"})
    assert out.retryable is False and out.error_details["code"] == "model_does_not_fit"
