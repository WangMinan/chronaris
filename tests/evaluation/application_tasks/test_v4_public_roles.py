from dataclasses import replace
from pathlib import Path

import pytest

from chronaris.evaluation.application_tasks.v4_public_data import public_subject_registry, DEVELOPMENT_SUBJECTS
from chronaris.representation import CheckpointRegistry, FoldLineage
from chronaris.representation.contracts import RepresentationContractError
from tests.representation.test_oof_export import _checkpoint


def test_public_subject_folds_exclude_development_from_confirmation(tmp_path):
    cogpilot, clare = tmp_path / "cogpilot", tmp_path / "clare"
    for index in range(1, 36):
        (cogpilot / f"sub-cp{index:03d}").mkdir(parents=True)
    for subject in (*DEVELOPMENT_SUBJECTS["clare"], *(str(i) for i in range(1000, 1015))):
        (clare / "EEG" / subject).mkdir(parents=True)
    registry = public_subject_registry(cogpilot_root=cogpilot, clare_root=clare)
    for domain, value in registry["domains"].items():
        development, confirmation = set(value["development_subjects"]), set(value["confirmation_subjects"])
        assert not development & confirmation
        for role, expected, count in (("development", development, 3), ("confirmation", confirmation, 5)):
            assert len(value["folds"][role]) == count
            scored = []
            for fold in value["folds"][role]:
                train, validation, held = (set(fold[name + "_subjects"]) for name in ("train", "validation", "held_out"))
                assert not (train & validation or train & held or validation & held)
                assert train | validation | held == expected
                scored.extend(held if role == "confirmation" else validation)
            assert set(scored) == expected and len(scored) == len(expected)
    assert registry == public_subject_registry(cogpilot_root=cogpilot, clare_root=clare)


def test_development_only_fold_is_explicit_and_registry_preserves_it(tmp_path):
    with pytest.raises(RepresentationContractError, match="non-empty"):
        FoldLineage("development", ("train",), ("validation",), ())
    fold = FoldLineage("development", ("train",), ("validation",), (), development_only=True)
    with pytest.raises(RepresentationContractError, match="no confirmation"):
        replace(fold, held_out_sample_ids=("confirmation",))
    record = replace(_checkpoint(tmp_path), fold=fold)
    registry_path = tmp_path / "registry.json"
    CheckpointRegistry(registry_path).register(record)
    restored = CheckpointRegistry(registry_path).require("chronaris", "development")
    assert restored.fold == fold
    assert restored.fold.to_dict()["development_only"] is True
