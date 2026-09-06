"""Re-evaluate existing engineering exports through the grouped frozen consumers."""
from dataclasses import replace
from pathlib import Path
import json
import time
import sys
import torch

from chronaris.evaluation.application_tasks.v4_development_data import load_development_inputs
from chronaris.evaluation.application_tasks.v4_grouped_consumers import native_consumer_context, run_native_method_consumers
from chronaris.representation import load_fusion_stream_batch

torch.set_num_threads(1)
root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("artifacts/application_evaluation/v4-grouped-consumer-reproduction")
root.mkdir(parents=True, exist_ok=True)
results = []
for domain in ("dingxin", "clare", "cogpilot"):
    started = time.perf_counter()
    _, _, fold, _, digest, targets, definitions, data = load_development_inputs(domain,
        "artifacts/application_evaluation/2026-09-06_v4-public-development",
        "docs/requirements/thesis-v4-public-subjects.json", smoke=True)
    context = native_consumer_context(domain, data, replace(fold, fold_id=fold.fold_id.removesuffix("__engineering_smoke")))
    source = Path("artifacts/application_evaluation") / ("2026-09-06_v4-real-domain-smoke" if domain == "dingxin"
        else "2026-09-06_v4-real-domain-smoke-repair") / domain / "all"
    for route in ("self_supervised", "task_guided"):
        export = source / (route + "_representations")
        if route == "task_guided":
            export = export / "chronaris"
        outputs = {role: load_fusion_stream_batch(export / role) for role in ("train", "validation")}
        assert outputs["train"].sample_ids == fold.train_sample_ids
        result = run_native_method_consumers(outputs=outputs, targets=targets, definitions=definitions, context=context,
            output_root=root / domain / route, label_used_for_encoder_training=route == "task_guided")
        results.append({"domain": domain, "route": route, "data_manifest_sha256": digest,
            "source_scope": "existing_engineering_subset", "result": result})
        print(domain, route, "completed", round(time.perf_counter() - started, 2), flush=True)
        (root / "summary.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
