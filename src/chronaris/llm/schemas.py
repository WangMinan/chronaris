"""Structured contracts for audited LLM preprocessing calls."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from typing import Mapping, Sequence

PROMPT_VERSION = "task_eval_llm_preprocessing.agent_guardrails.v2"
SCHEMA_VERSION = "task_eval_llm_preprocessing_context.v2"

ALLOWED_FIELD_ROLES = {
    "attitude",
    "event_replay",
    "physiological_load",
    "position",
    "risk_signal",
    "schema_gap",
    "signal_quality",
    "speed",
    "unknown",
    "vertical",
}
ALLOWED_REVIEW_DECISIONS = {
    "keep_current_rule",
    "needs_human_review",
    "revise_current_rule",
}
ALLOWED_SCHEMA_POLICIES = {
    "canonical_fill_nan",
    "drop_or_mask",
    "human_review_required",
    "source_requery_required",
}
ALLOWED_SEMANTIC_RECIPES = {
    "coordination_gap",
    "gap_plus_event",
    "physiology_plus_gap",
    "vehicle_plus_event",
}


@dataclass(frozen=True, slots=True)
class LLMMessage:
    """One chat message sent to an OpenAI-compatible provider."""

    role: str
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass(frozen=True, slots=True)
class LLMTaskRequest:
    """One audited preprocessing request."""

    request_id: str
    run_id: str
    task_name: str
    provider: str
    model: str
    prompt_version: str
    output_schema_version: str
    input_payload: Mapping[str, object]
    messages: tuple[LLMMessage, ...]
    input_hash: str = ""

    def with_hash(self) -> "LLMTaskRequest":
        if self.input_hash:
            return self
        return LLMTaskRequest(
            request_id=self.request_id,
            run_id=self.run_id,
            task_name=self.task_name,
            provider=self.provider,
            model=self.model,
            prompt_version=self.prompt_version,
            output_schema_version=self.output_schema_version,
            input_payload=self.input_payload,
            messages=self.messages,
            input_hash=stable_input_hash(self.input_payload),
        )

    def to_audit_dict(self) -> dict[str, object]:
        resolved = self.with_hash()
        return {
            "request_id": resolved.request_id,
            "run_id": resolved.run_id,
            "task_name": resolved.task_name,
            "provider": resolved.provider,
            "model": resolved.model,
            "prompt_version": resolved.prompt_version,
            "output_schema_version": resolved.output_schema_version,
            "input_hash": resolved.input_hash,
            "messages": [message.to_dict() for message in resolved.messages],
            "input_payload": resolved.input_payload,
        }


@dataclass(frozen=True, slots=True)
class LLMTaskResponse:
    """One provider response with timing and error metadata."""

    request_id: str
    provider: str
    model: str
    status: str
    content: str | None = None
    parsed_payload: Mapping[str, object] | None = None
    raw_response: Mapping[str, object] | None = None
    latency_ms: float | None = None
    retry_count: int = 0
    token_usage: Mapping[str, object] = field(default_factory=dict)
    error_summary: str | None = None

    def to_audit_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["parsed_payload"] = dict(self.parsed_payload or {})
        payload["raw_response"] = dict(self.raw_response or {})
        payload["token_usage"] = dict(self.token_usage or {})
        return payload


def json_dumps_stable(payload: Mapping[str, object] | Sequence[object]) -> str:
    """Serialize payloads consistently for hashing and audit files."""

    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_input_hash(payload: Mapping[str, object] | Sequence[object]) -> str:
    """Return a stable SHA-256 hash for a preprocessing input payload."""

    return hashlib.sha256(json_dumps_stable(payload).encode("utf-8")).hexdigest()


def extract_json_payload(text: str | None) -> dict[str, object]:
    """Extract one JSON object from a provider text response."""

    if not text:
        return {}
    stripped = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, flags=re.DOTALL)
    if fenced:
        stripped = fenced.group(1)
    if not stripped.startswith("{"):
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start >= 0 and end >= start:
            stripped = stripped[start : end + 1]
    return json.loads(stripped)


def coerce_field_semantics(payload: Mapping[str, object]) -> list[dict[str, object]]:
    """Validate field semantic rows while keeping the output usable downstream."""

    rows = payload.get("field_semantics") or payload.get("rows") or []
    if isinstance(rows, Mapping):
        rows = [
            {"feature_name": feature_name, **dict(row_payload)}
            for feature_name, row_payload in rows.items()
            if isinstance(row_payload, Mapping)
        ]
    if not isinstance(rows, list):
        return []
    coerced = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        feature_name = _as_text(row.get("feature_name"))
        if not feature_name:
            continue
        stream_kind = _coerce_choice(
            _as_text(row.get("stream_kind")),
            {"physiology", "vehicle"},
            _infer_stream_kind(feature_name),
        )
        role = _coerce_field_role(
            _as_text(row.get("llm_semantic_role"))
            or _as_text(row.get("measurement_role"))
            or _as_text(row.get("semantic_role")),
            feature_name=feature_name,
            stream_kind=stream_kind,
        )
        confidence = _coerce_confidence(row.get("confidence"))
        coerced.append(
            {
                "stream_kind": stream_kind,
                "measurement_group": _as_text(row.get("measurement_group")) or _measurement_group(feature_name),
                "feature_name": feature_name,
                "llm_semantic_role": role,
                "llm_unit_guess": _as_text(row.get("llm_unit_guess")) or _as_text(row.get("unit")),
                "evidence": _as_text(row.get("evidence")) or "schema_card",
                "confidence": confidence,
                "needs_human_review": bool(row.get("needs_human_review", confidence < 0.65)),
            }
        )
    return coerced


def coerce_weak_label_review(payload: Mapping[str, object]) -> list[dict[str, object]]:
    """Validate task-level weak-label review rows."""

    rows = payload.get("weak_label_rule_review") or payload.get("rows") or []
    if not isinstance(rows, list):
        return []
    coerced = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        task_name = _as_text(row.get("task_name"))
        if not task_name:
            continue
        decision = _coerce_choice(
            _as_text(row.get("review_decision")),
            ALLOWED_REVIEW_DECISIONS,
            "needs_human_review",
        )
        confidence = _coerce_confidence(row.get("confidence"))
        coerced.append(
            {
                "task_name": task_name,
                "review_decision": decision,
                "confidence": confidence,
                "agreement_basis": _as_text(row.get("agreement_basis")) or "rule_summary",
                "conflict_basis": _as_text(row.get("conflict_basis")),
                "needs_human_review": bool(row.get("needs_human_review", decision != "keep_current_rule")),
                "recommended_action": _as_text(row.get("recommended_action")) or "audit_before_training",
            }
        )
    return coerced


def coerce_semantic_query_hints(payload: Mapping[str, object]) -> list[dict[str, object]]:
    """Validate LLM semantic query hints through the recipe whitelist."""

    rows = payload.get("semantic_query_hints") or payload.get("rows") or []
    if isinstance(rows, Mapping):
        rows = [
            {"name": key, **value} if isinstance(value, Mapping) else {"name": key, "recipe": value}
            for key, value in rows.items()
        ]
    if not isinstance(rows, list):
        return []
    coerced = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        recipe = _as_text(row.get("recipe"))
        name = _safe_identifier(_as_text(row.get("name")) or _as_text(row.get("query_name")))
        if not name and recipe in ALLOWED_SEMANTIC_RECIPES:
            name = f"llm_{recipe}"
        if not name or recipe not in ALLOWED_SEMANTIC_RECIPES:
            continue
        coerced.append(
            {
                "name": name,
                "recipe": recipe,
                "confidence": _coerce_confidence(row.get("confidence")),
                "source": _as_text(row.get("source")) or "llm_preprocessing_context",
                "needs_human_review": bool(row.get("needs_human_review", False)),
            }
        )
    return coerced


def coerce_schema_gap_policy(payload: Mapping[str, object]) -> dict[str, object]:
    """Validate schema-gap policy proposals without allowing fabricated values."""

    policy = payload.get("schema_gap_policy") if "schema_gap_policy" in payload else payload
    if isinstance(policy, list):
        rows = policy
        policy = {}
    elif isinstance(policy, Mapping):
        rows = policy.get("policies") or policy.get("rules") or []
    else:
        return {}
    if not rows and all(isinstance(key, str) for key in policy):
        rows = [
            {
                "measurement_group": key,
                "policy_type": value,
                "reason": "LLM shorthand schema-gap policy",
            }
            for key, value in policy.items()
            if key.startswith("BUS")
        ]
    if not isinstance(rows, list):
        rows = []
    coerced_rows = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        policy_type = _coerce_choice(
            _as_text(row.get("policy_type")) or _as_text(row.get("policy")),
            ALLOWED_SCHEMA_POLICIES,
            "human_review_required",
        )
        coerced_rows.append(
            {
                "measurement_group": _as_text(row.get("measurement_group")) or "unknown",
                "policy_type": policy_type,
                "reason": _as_text(row.get("reason")) or "schema gap requires audited handling",
                "allows_value_fabrication": False,
                "needs_human_review": bool(row.get("needs_human_review", policy_type == "human_review_required")),
            }
        )
    return {
        "status": _as_text(policy.get("status")) or "draft_policy",
        "canonical_exact_boundary": _as_text(policy.get("canonical_exact_boundary"))
        or "canonical exact is a service payload contract, not native exact evidence",
        "native_exact_claim_allowed": False,
        "policies": coerced_rows,
    }


def coerce_runtime_explanations(payload: Mapping[str, object]) -> list[dict[str, object]]:
    """Validate runtime case explanations."""

    rows = payload.get("runtime_case_explanations") or payload.get("rows") or []
    if not isinstance(rows, list):
        return []
    coerced = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        sample_id = _as_text(row.get("sample_id"))
        if not sample_id:
            continue
        coerced.append(
            {
                "sample_id": sample_id,
                "model_prediction": _as_text(row.get("model_prediction")),
                "semantic_attribution": _as_text(row.get("semantic_attribution")),
                "schema_gap_note": _as_text(row.get("schema_gap_note")),
                "weak_label_boundary": _as_text(row.get("weak_label_boundary"))
                or "weak_label_proxy_not_manual_ground_truth",
                "source_paths": list(row.get("source_paths") or []),
                "confidence": _coerce_confidence(row.get("confidence")),
                "needs_human_review": bool(row.get("needs_human_review", True)),
            }
        )
    return coerced


def _as_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _coerce_confidence(value: object) -> float:
    try:
        resolved = float(value)
    except (TypeError, ValueError):
        return 0.5
    return max(0.0, min(1.0, resolved))


def _coerce_choice(value: str | None, allowed: set[str], fallback: str) -> str:
    if value in allowed:
        return value
    return fallback


def _coerce_field_role(value: str | None, *, feature_name: str, stream_kind: str) -> str:
    normalized = (value or "").strip().lower()
    if normalized in ALLOWED_FIELD_ROLES:
        return normalized
    if stream_kind == "physiology":
        if "quality" in normalized:
            return "signal_quality"
        return "physiological_load"
    lowered = feature_name.lower()
    if any(token in normalized or token in lowered for token in ("attitude", "pitch", "roll", "yaw", "heading")):
        return "attitude"
    if any(token in normalized or token in lowered for token in ("speed", "velocity")):
        return "speed"
    if any(token in normalized or token in lowered for token in ("vertical", "altitude", "height")):
        return "vertical"
    if any(token in normalized or token in lowered for token in ("position", "lat", "lon")):
        return "position"
    return "risk_signal" if feature_name.startswith("BUS") else "unknown"


def _infer_stream_kind(feature_name: str) -> str:
    lowered = feature_name.lower()
    if lowered.startswith(("eeg.", "spo2.", "hr.", "heart.")):
        return "physiology"
    return "vehicle"


def _measurement_group(feature_name: str) -> str:
    if "." in feature_name:
        return feature_name.split(".", 1)[0]
    return feature_name


def _safe_identifier(value: str | None) -> str | None:
    if not value:
        return None
    lowered = value.strip().lower().replace("-", "_").replace(" ", "_")
    if not re.match(r"^[a-z][a-z0-9_]{1,63}$", lowered):
        return None
    return lowered
