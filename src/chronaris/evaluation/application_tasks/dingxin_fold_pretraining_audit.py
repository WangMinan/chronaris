"""Acceptance checks for one real Dingxin fold representation smoke."""

from __future__ import annotations


def build_dingxin_fold_pretraining_acceptance_rows(
    *,
    fold,
    normalizer,
    training_results,
    resumed_training_results,
    registry,
    initial_exports,
    resumed_exports,
    alignment_hashes,
    naive_manifest,
    maximum_rss_mb,
):
    role_counts = {
        "train": len(fold.train_sample_ids),
        "validation": len(fold.validation_sample_ids),
        "held_out": len(fold.held_out_sample_ids),
    }
    fit_ids = set(normalizer.fit_sample_ids)
    non_train = set(fold.validation_sample_ids + fold.held_out_sample_ids)
    return [
        _check("primary_fold_role_counts", role_counts == {"train": 31, "validation": 31, "held_out": 31}, role_counts, {"train": 31, "validation": 31, "held_out": 31}),
        _check("normalizer_inner_train_only", fit_ids == set(fold.train_sample_ids) and not fit_ids & non_train, len(fit_ids), 31),
        _check("five_trainable_checkpoints", len(training_results) == 5 and all(result.status in {"completed", "resumed"} for result in training_results), [result.status for result in training_results], "five complete or resumed"),
        _check("one_epoch_has_31_steps", all(result.step_count == 31 for result in training_results), [result.step_count for result in training_results], "31 each"),
        _check("five_checkpoint_resume", len(resumed_training_results) == 5 and all(result.status == "resumed" for result in resumed_training_results), [result.status for result in resumed_training_results], "five resumed"),
        _check("six_checkpoint_records", len(registry.records) == 6, len(registry.records), 6),
        _check("encoder_training_has_no_labels", all(not record.label_used_for_encoder_training for record in registry.records.values()), False, False),
        _check("naive_randomized_train_only_pca", naive_manifest["pca_projector"]["solver"] == "randomized" and set(naive_manifest["pca_projector"]["fit_sample_ids"]) == set(fold.train_sample_ids), naive_manifest["pca_projector"]["solver"], "randomized on inner-train"),
        _check("eighteen_role_exports", len(initial_exports) == 18, len(initial_exports), 18),
        _check("eighteen_exports_resume", len(resumed_exports) == 18 and all(result.status == "resumed" for result in resumed_exports), sum(result.status == "resumed" for result in resumed_exports), 18),
        _check("six_method_role_alignment", set(alignment_hashes) == {"train", "validation", "held_out"} and all(len(value) == 64 for value in alignment_hashes.values()), alignment_hashes, "three role hashes"),
        _check("bounded_peak_memory", maximum_rss_mb < 2500.0, round(maximum_rss_mb, 1), "<2500 MB"),
    ]


def _check(check_id, passed, actual, expected):
    return {
        "check_id": check_id,
        "passed": bool(passed),
        "actual": actual,
        "expected": expected,
    }
