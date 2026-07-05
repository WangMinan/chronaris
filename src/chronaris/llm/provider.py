"""LLM provider implementations for task evaluation preprocessing."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from typing import Mapping

from chronaris.llm.schemas import (
    LLMTaskRequest,
    LLMTaskResponse,
    extract_json_payload,
)

DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com/chat/completions"


class LLMProvider(ABC):
    """Provider boundary used by the task evaluation preprocessing pipeline."""

    provider_name: str
    model: str

    @abstractmethod
    def generate(self, request: LLMTaskRequest) -> LLMTaskResponse:
        """Return one audited provider response."""


class DeepSeekChatProvider(LLMProvider):
    """Minimal OpenAI-compatible DeepSeek chat provider."""

    provider_name = "deepseek"

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None,
        base_url: str = DEFAULT_DEEPSEEK_BASE_URL,
        timeout_seconds: float = 60.0,
        max_retries: int = 1,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries

    def generate(self, request: LLMTaskRequest) -> LLMTaskResponse:
        if not self.api_key:
            return LLMTaskResponse(
                request_id=request.request_id,
                provider=self.provider_name,
                model=self.model,
                status="expected_failure",
                error_summary="missing DEEPSEEK_API_KEY",
                retry_count=0,
            )
        payload = {
            "model": self.model,
            "messages": [message.to_dict() for message in request.messages],
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        retry_count = 0
        started_at = time.perf_counter()
        last_error: str | None = None
        while retry_count <= self.max_retries:
            try:
                raw_response = _post_json(
                    self.base_url,
                    payload=payload,
                    headers=headers,
                    timeout_seconds=self.timeout_seconds,
                )
                latency_ms = (time.perf_counter() - started_at) * 1000.0
                content = _extract_chat_content(raw_response)
                parsed = extract_json_payload(content)
                return LLMTaskResponse(
                    request_id=request.request_id,
                    provider=self.provider_name,
                    model=self.model,
                    status="success",
                    content=content,
                    parsed_payload=parsed,
                    raw_response=_redact_raw_response(raw_response),
                    latency_ms=latency_ms,
                    retry_count=retry_count,
                    token_usage=dict(raw_response.get("usage") or {}),
                )
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                last_error = str(exc)
                if retry_count >= self.max_retries:
                    break
                retry_count += 1
                time.sleep(min(2.0, 0.5 * (retry_count + 1)))
        latency_ms = (time.perf_counter() - started_at) * 1000.0
        return LLMTaskResponse(
            request_id=request.request_id,
            provider=self.provider_name,
            model=self.model,
            status="expected_failure",
            latency_ms=latency_ms,
            retry_count=retry_count,
            error_summary=last_error or "DeepSeek request failed",
        )


class MockLLMProvider(LLMProvider):
    """Deterministic provider for offline tests and no-network dry runs."""

    provider_name = "mock"

    def __init__(self, *, model: str = "mock-task-eval-llm") -> None:
        self.model = model

    def generate(self, request: LLMTaskRequest) -> LLMTaskResponse:
        started_at = time.perf_counter()
        payload = self._mock_payload(request)
        content = json.dumps(payload, ensure_ascii=False)
        return LLMTaskResponse(
            request_id=request.request_id,
            provider=self.provider_name,
            model=self.model,
            status="success",
            content=content,
            parsed_payload=payload,
            raw_response={"mock": True},
            latency_ms=(time.perf_counter() - started_at) * 1000.0,
            retry_count=0,
            token_usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        )

    def _mock_payload(self, request: LLMTaskRequest) -> dict[str, object]:
        if request.task_name == "field_semantics":
            fields = request.input_payload.get("fields", [])
            rows = []
            for field in fields if isinstance(fields, list) else []:
                if not isinstance(field, Mapping):
                    continue
                feature_name = str(field.get("feature_name", ""))
                stream_kind = str(field.get("stream_kind", "vehicle"))
                rows.append(
                    {
                        "stream_kind": stream_kind,
                        "measurement_group": field.get("measurement_group"),
                        "feature_name": feature_name,
                        "llm_semantic_role": _mock_role(feature_name, stream_kind),
                        "llm_unit_guess": None,
                        "evidence": "mock schema_card feature-name heuristic",
                        "confidence": 0.82 if stream_kind == "physiology" else 0.68,
                        "needs_human_review": stream_kind == "vehicle",
                    }
                )
            return {"field_semantics": rows}
        if request.task_name == "weak_label_review":
            tasks = request.input_payload.get("task_summaries", [])
            rows = []
            for task in tasks if isinstance(tasks, list) else []:
                if not isinstance(task, Mapping):
                    continue
                task_name = str(task.get("task_name"))
                rows.append(
                    {
                        "task_name": task_name,
                        "review_decision": "keep_current_rule",
                        "confidence": 0.74 if task_name != "event_replay_tag" else 0.64,
                        "agreement_basis": "mock review preserves current weak-label rule",
                        "conflict_basis": None,
                        "needs_human_review": task_name == "event_replay_tag",
                        "recommended_action": "keep_as_preprocessing_context_only",
                    }
                )
            return {"weak_label_rule_review": rows}
        if request.task_name == "semantic_query_hints":
            return {
                "semantic_query_hints": [
                    {
                        "name": "llm_schema_gap_review",
                        "recipe": "coordination_gap",
                        "confidence": 0.7,
                        "source": "mock_schema_gap_and_runtime_case_card",
                        "needs_human_review": True,
                    }
                ]
            }
        if request.task_name == "schema_gap_policy":
            groups = request.input_payload.get("missing_measurement_groups", {})
            policies = []
            for group in sorted(groups) if isinstance(groups, Mapping) else []:
                policies.append(
                    {
                        "measurement_group": group,
                        "policy_type": "canonical_fill_nan",
                        "reason": "missing BUS group must be explicit in canonical payload",
                        "allows_value_fabrication": False,
                        "needs_human_review": False,
                    }
                )
            return {
                "schema_gap_policy": {
                    "status": "draft_policy",
                    "canonical_exact_boundary": (
                        "canonical exact is a service payload contract, not native exact evidence"
                    ),
                    "native_exact_claim_allowed": False,
                    "policies": policies,
                }
            }
        if request.task_name == "runtime_explanations":
            cases = request.input_payload.get("runtime_cases", [])
            rows = []
            for case in cases if isinstance(cases, list) else []:
                if not isinstance(case, Mapping):
                    continue
                rows.append(
                    {
                        "sample_id": case.get("sample_id"),
                        "model_prediction": f"risk={case.get('risk_proxy_prediction')}",
                        "semantic_attribution": f"top_query={case.get('semantic_top_query_name')}",
                        "schema_gap_note": (
                            f"native={case.get('native_feature_schema_status')}, "
                            f"canonical={case.get('canonical_feature_schema_status')}"
                        ),
                        "weak_label_boundary": "weak_label_proxy_not_manual_ground_truth",
                        "source_paths": [
                            case.get("source_path"),
                            case.get("support_source_path"),
                            case.get("runtime_schema_contract_source_path"),
                        ],
                        "confidence": 0.66,
                        "needs_human_review": True,
                    }
                )
            return {"runtime_case_explanations": rows}
        return {}


def resolve_llm_provider(
    *,
    provider_name: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> LLMProvider:
    """Resolve a provider from explicit arguments and environment defaults."""

    resolved_provider = (provider_name or os.environ.get("CHRONARIS_LLM_PROVIDER") or "deepseek").lower()
    resolved_model = model or os.environ.get("CHRONARIS_LLM_MODEL") or "deepseek-v4-pro"
    if resolved_provider == "mock":
        return MockLLMProvider(model=resolved_model)
    if resolved_provider == "deepseek":
        return DeepSeekChatProvider(
            model=resolved_model,
            api_key=api_key if api_key is not None else os.environ.get("DEEPSEEK_API_KEY"),
            base_url=base_url or os.environ.get("CHRONARIS_LLM_BASE_URL") or DEFAULT_DEEPSEEK_BASE_URL,
        )
    raise ValueError(f"unsupported LLM provider: {resolved_provider}")


def _post_json(
    url: str,
    *,
    payload: Mapping[str, object],
    headers: Mapping[str, str],
    timeout_seconds: float,
) -> dict[str, object]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=dict(headers), method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise OSError(f"HTTP {exc.code}: {body[:500]}") from exc


def _extract_chat_content(raw_response: Mapping[str, object]) -> str:
    choices = raw_response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("provider response did not include choices")
    first = choices[0]
    if not isinstance(first, Mapping):
        raise ValueError("provider response choice is not an object")
    message = first.get("message")
    if not isinstance(message, Mapping):
        raise ValueError("provider response choice has no message")
    content = message.get("content")
    if not isinstance(content, str):
        raise ValueError("provider response message has no text content")
    return content


def _redact_raw_response(raw_response: Mapping[str, object]) -> dict[str, object]:
    return {
        "id": raw_response.get("id"),
        "object": raw_response.get("object"),
        "created": raw_response.get("created"),
        "model": raw_response.get("model"),
        "usage": raw_response.get("usage"),
    }


def _mock_role(feature_name: str, stream_kind: str) -> str:
    lowered = feature_name.lower()
    if stream_kind == "physiology":
        return "physiological_load"
    if any(token in lowered for token in ("pitch", "roll", "yaw", "heading")):
        return "attitude"
    if any(token in lowered for token in ("speed", "velocity")):
        return "speed"
    if any(token in lowered for token in ("alt", "vertical")):
        return "vertical"
    if "bus" in lowered:
        return "risk_signal"
    return "unknown"
