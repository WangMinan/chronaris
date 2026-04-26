"""Stage H export and partial-data tests."""

from __future__ import annotations

import json
import sys
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path
import unittest

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chronaris.access.stage_h_profile import StageHProfileResolver, StageHSortieProfile, StageHViewProfile
from chronaris.evaluation import evaluate_alignment_projection_thresholds, summarize_alignment_projection_diagnostics
from chronaris.pipelines.alignment_preview import (
    AlignmentPreviewIntermediateExport,
    AlignmentPreviewSampleIntermediate,
    StreamIntermediateSnapshot,
)
from chronaris.pipelines.causal_fusion import (
    StageGCausalFusionConfig,
    StageGCausalFusionResult,
    StageGCausalFusionSample,
    StageGCausalFusionTensorExport,
)
from chronaris.pipelines.partial_data import PartialDataBuilder, PartialDataConfig, PartialDataEntry
from chronaris.pipelines.stage_h_export import (
    StageHExportConfig,
    StageHExportPipeline,
    StageHViewExecutionResult,
)
from chronaris.schema.models import (
    AlignedPoint,
    AlignedSortieBundle,
    DatasetBuildResult,
    RawPoint,
    SampleWindow,
    SortieLocator,
    SortieMetadata,
    StreamKind,
)
from chronaris.schema.real_bus import CollectTaskMetadata, FlightTaskMetadata, StorageAnalysis


def _utc(year: int, month: int, day: int, hour: int, minute: int, second: int) -> datetime:
    return datetime(year, month, day, hour, minute, second, tzinfo=timezone.utc)


def _aligned_point(stream_kind: StreamKind, measurement: str, second: int, values: dict[str, object]) -> AlignedPoint:
    return AlignedPoint(
        point=RawPoint(
            stream_kind=stream_kind,
            measurement=measurement,
            timestamp=_utc(2025, 10, 5, 1, 35, second),
            values=values,
        ),
        offset_ms=second * 1000,
    )


def _stream_snapshot(
    *,
    feature_names: tuple[str, ...],
    projected_rows: tuple[tuple[float, ...], ...],
    hidden_rows: tuple[tuple[float, ...], ...],
    offsets_s: tuple[float, ...],
) -> StreamIntermediateSnapshot:
    return StreamIntermediateSnapshot(
        feature_names=feature_names,
        point_count=len(offsets_s),
        observation_offsets_s=offsets_s,
        reference_offsets_s=offsets_s,
        observation_hidden_states=hidden_rows,
        reference_hidden_states=hidden_rows,
        reference_projected_states=projected_rows,
        mean_observation_hidden_l2=1.0,
        mean_reference_hidden_l2=1.0,
        mean_reference_projection_l2=1.0,
    )


def _intermediate_export(sample_prefix: str) -> AlignmentPreviewIntermediateExport:
    sample = AlignmentPreviewSampleIntermediate(
        sample_id=f"{sample_prefix}:0000",
        physiology=_stream_snapshot(
            feature_names=("eeg.af3", "spo2.spo2"),
            projected_rows=((1.0, 0.0), (0.0, 1.0)),
            hidden_rows=((0.5, 0.1), (0.2, 0.7)),
            offsets_s=(0.0, 5.0),
        ),
        vehicle=_stream_snapshot(
            feature_names=("BUS001.speed", "BUS001.accel"),
            projected_rows=((1.0, 0.0), (0.0, 1.0)),
            hidden_rows=((0.4, 0.2), (0.1, 0.8)),
            offsets_s=(0.0, 5.0),
        ),
        mean_reference_projection_cosine=1.0,
    )
    return AlignmentPreviewIntermediateExport(
        partition="test",
        sample_count=1,
        reference_point_count=2,
        samples=(sample,),
    )


def _dataset_result(sortie_id: str) -> DatasetBuildResult:
    return DatasetBuildResult(
        aligned_bundle=AlignedSortieBundle(
            locator=SortieLocator(sortie_id=sortie_id),
            metadata=SortieMetadata(sortie_id=sortie_id),
            reference_time=_utc(2025, 10, 5, 1, 35, 0),
        ),
        windows=(
            SampleWindow(
                sample_id=f"{sortie_id}:0000",
                sortie_id=sortie_id,
                window_index=0,
                start_offset_ms=0,
                end_offset_ms=5000,
                physiology_points=(
                    _aligned_point(StreamKind.PHYSIOLOGY, "eeg", 0, {"af3": 1.0}),
                ),
                vehicle_points=(
                    _aligned_point(StreamKind.VEHICLE, "BUS001", 0, {"speed": 200.0}),
                ),
            ),
            SampleWindow(
                sample_id=f"{sortie_id}:0001",
                sortie_id=sortie_id,
                window_index=1,
                start_offset_ms=5000,
                end_offset_ms=10000,
                physiology_points=(
                    _aligned_point(StreamKind.PHYSIOLOGY, "eeg", 5, {"af3": 2.0}),
                ),
                vehicle_points=(
                    _aligned_point(StreamKind.VEHICLE, "BUS001", 5, {"speed": 201.0}),
                ),
            ),
        ),
    )


def _stage_g_result(sample_id: str) -> tuple[StageGCausalFusionResult, StageGCausalFusionTensorExport]:
    config = StageGCausalFusionConfig(state_source="hidden")
    result = StageGCausalFusionResult(
        config=config,
        partition="test",
        sample_count=1,
        reference_point_count=2,
        state_dim=2,
        fused_dim=6,
        mean_attention_entropy=0.1,
        mean_max_attention=0.9,
        mean_causal_option_count=1.5,
        mean_top_event_score=0.8,
        mean_top_contribution_score=0.7,
        samples=(
            StageGCausalFusionSample(
                sample_id=sample_id,
                reference_point_count=2,
                state_dim=2,
                fused_dim=6,
                mean_attention_entropy=0.1,
                mean_max_attention=0.9,
                mean_causal_option_count=1.5,
                top_event_offset_s=5.0,
                top_event_score=0.8,
                top_contribution_offset_s=5.0,
                top_contribution_score=0.7,
                attention_weights=((1.0, 0.0), (0.5, 0.5)),
                vehicle_event_scores=(0.0, 1.0),
            ),
        ),
    )
    tensor_export = StageGCausalFusionTensorExport(
        sample_ids=(sample_id,),
        fused_states=(((1.0, 0.0, 0.5, 0.1, 0.5, -0.1), (0.0, 1.0, 0.2, 0.8, -0.2, 0.2)),),
        attention_weights=(((1.0, 0.0), (0.5, 0.5)),),
        vehicle_event_scores=((0.0, 1.0),),
    )
    return result, tensor_export


def _view_execution(sortie_id: str, *, include_stage_g: bool) -> StageHViewExecutionResult:
    intermediate_export = _intermediate_export(sortie_id)
    diagnostics = summarize_alignment_projection_diagnostics(intermediate_export)
    threshold = evaluate_alignment_projection_thresholds(diagnostics)
    stage_g_result = None
    stage_g_tensor = None
    if include_stage_g:
        stage_g_result, stage_g_tensor = _stage_g_result(f"{sortie_id}:0000")
    return StageHViewExecutionResult(
        dataset_result=_dataset_result(sortie_id),
        sample_ids=(f"{sortie_id}:0000",),
        split_summary={"train": 1, "validation": 0, "test": 1},
        train_metrics={"total": 1.0},
        validation_metrics={"total": 1.0},
        test_metrics={"total": 1.0},
        intermediate_export=intermediate_export,
        diagnostics_summary=diagnostics,
        threshold_evaluation=threshold,
        stage_g_result=stage_g_result,
        stage_g_tensor_export=stage_g_tensor,
        vehicle_field_metadata={"status": "loaded", "field_count": 4},
    )


class _FakeFlightTaskReader:
    def fetch_by_locator(self, locator: SortieLocator) -> FlightTaskMetadata:
        if locator.sortie_id == "20251002_单01_ACT-8_翼云_J16_12#01":
            return FlightTaskMetadata(
                flight_task_id=6000002,
                sortie_number=locator.sortie_id,
                flight_batch_id=6000002,
                batch_number="20251002_单01",
                flight_date=date(2025, 10, 2),
                mission_code="ACT-8",
                aircraft_model="J16",
                aircraft_number="12",
                up_pilot_id=10035,
                down_pilot_id=10033,
                source_sortie_id=None,
            )
        return FlightTaskMetadata(
            flight_task_id=6000001,
            sortie_number=locator.sortie_id,
            flight_batch_id=6000001,
            batch_number="20251005_四01",
            flight_date=date(2025, 10, 5),
            mission_code="ACT-4",
            aircraft_model="J20",
            aircraft_number="22",
            up_pilot_id=10033,
            source_sortie_id="2100448-10033",
        )


class _FakeCollectTaskReader:
    def fetch_for_flight_task(self, flight_task: FlightTaskMetadata) -> CollectTaskMetadata:
        if flight_task.sortie_number == "20251002_单01_ACT-8_翼云_J16_12#01":
            return CollectTaskMetadata(
                collect_task_id=2100450,
                collect_date=date(2025, 10, 2),
                collect_start_time=_utc(2025, 10, 2, 7, 45, 0).replace(tzinfo=None),
                collect_end_time=_utc(2025, 10, 2, 12, 21, 38).replace(tzinfo=None),
            )
        return CollectTaskMetadata(
            collect_task_id=2100448,
            collect_date=date(2025, 10, 5),
            collect_start_time=_utc(2025, 10, 5, 9, 35, 0).replace(tzinfo=None),
            collect_end_time=_utc(2025, 10, 5, 9, 38, 1).replace(tzinfo=None),
        )


class _FakeStorageAnalysisReader:
    def list_for_sortie(self, locator: SortieLocator, *, category: str | None = None) -> tuple[StorageAnalysis, ...]:
        if locator.sortie_id == "20251002_单01_ACT-8_翼云_J16_12#01":
            measurements = tuple(f"BUS600001911002{index}" for index in range(1, 7))
            analysis_base = 6000019110020
        else:
            measurements = (
                "BUS6000019110015",
                "BUS6000019110016",
                "BUS6000019110017",
                "BUS6000019110018",
                "BUS6000019110019",
                "BUS6000019110020",
            )
            analysis_base = 6000019110014
        return tuple(
            StorageAnalysis(
                analysis_id=analysis_base + index,
                category=category or "BUS",
                bucket="bus",
                measurement=measurement,
                sortie_number=locator.sortie_id,
            )
            for index, measurement in enumerate(measurements, start=1)
        )


class _FakeDistinctMeasurementReader:
    def fetch_measurements(self, **_: object) -> tuple[str, ...]:
        return (
            "eeg",
            "spo2",
            "tshirt_ecg_accel_gyro",
        )


class _FakeProfileResolver:
    def __init__(self, profiles: tuple[StageHSortieProfile, ...]) -> None:
        self._profiles = profiles

    def resolve_many(self, sortie_ids: tuple[str, ...]) -> tuple[StageHSortieProfile, ...]:
        return tuple(profile for profile in self._profiles if profile.sortie_id in sortie_ids)


class _FakeViewRunner:
    def __init__(self, include_stage_g: bool = True) -> None:
        self.include_stage_g = include_stage_g

    def run(
        self,
        profile: StageHSortieProfile,
        view: StageHViewProfile,
        *,
        export_start_utc: datetime,
        export_stop_utc: datetime,
    ) -> StageHViewExecutionResult:
        del view, export_start_utc, export_stop_utc
        return _view_execution(profile.sortie_id, include_stage_g=self.include_stage_g)


def _profile(sortie_id: str, pilot_ids: tuple[int, ...]) -> StageHSortieProfile:
    flight_task = FlightTaskMetadata(
        flight_task_id=1 if sortie_id.endswith("22#01") else 2,
        sortie_number=sortie_id,
        flight_batch_id=1,
        batch_number="batch",
        flight_date=date(2025, 10, 5),
        mission_code="ACT",
        aircraft_model="JXX",
        aircraft_number="01",
        up_pilot_id=pilot_ids[0],
        down_pilot_id=pilot_ids[1] if len(pilot_ids) > 1 else None,
        source_sortie_id=None if len(pilot_ids) > 1 else f"2100448-{pilot_ids[0]}",
    )
    collect_task = CollectTaskMetadata(
        collect_task_id=2100448 if len(pilot_ids) == 1 else 2100450,
        collect_date=date(2025, 10, 5),
        collect_start_time=_utc(2025, 10, 5, 9, 35, 0).replace(tzinfo=None),
        collect_end_time=_utc(2025, 10, 5, 9, 38, 1).replace(tzinfo=None),
    )
    return StageHSortieProfile(
        sortie_id=sortie_id,
        flight_task=flight_task,
        collect_task=collect_task,
        pilot_ids=pilot_ids,
        views=tuple(
            StageHViewProfile(
                view_id=f"{sortie_id}__pilot_{pilot_id}",
                pilot_id=pilot_id,
            )
            for pilot_id in pilot_ids
        ),
        physiology_bucket="physiological_input",
        vehicle_bucket="bus",
        available_physiology_measurements=("eeg", "spo2", "tshirt_ecg_accel_gyro"),
        model_physiology_measurements=("eeg", "spo2"),
        vehicle_measurements=("BUS001", "BUS002"),
        vehicle_analysis_ids={"BUS001": 1, "BUS002": 2},
        clip_start_utc=_utc(2025, 10, 5, 1, 35, 0),
        clip_stop_utc=_utc(2025, 10, 5, 1, 38, 1),
        pilot_resolution_source="source_sortie_id",
    )


class StageHProfileResolverTest(unittest.TestCase):
    def test_resolver_returns_dual_pilot_and_single_pilot_profiles(self) -> None:
        resolver = StageHProfileResolver(
            flight_task_reader=_FakeFlightTaskReader(),
            collect_task_reader=_FakeCollectTaskReader(),
            storage_analysis_reader=_FakeStorageAnalysisReader(),
            distinct_measurement_reader=_FakeDistinctMeasurementReader(),
        )

        dual = resolver.resolve(SortieLocator(sortie_id="20251002_单01_ACT-8_翼云_J16_12#01"))
        single = resolver.resolve(SortieLocator(sortie_id="20251005_四01_ACT-4_云_J20_22#01"))

        self.assertEqual(tuple(view.pilot_id for view in dual.views), (10035, 10033))
        self.assertEqual(len(dual.vehicle_measurements), 6)
        self.assertEqual(dual.vehicle_measurements[0], "BUS6000019110021")
        self.assertEqual(tuple(view.pilot_id for view in single.views), (10033,))
        self.assertEqual(len(single.vehicle_measurements), 6)
        self.assertEqual(single.vehicle_measurements[-1], "BUS6000019110020")


class StageHExportPipelineTest(unittest.TestCase):
    def test_pipeline_writes_stage_h_artifacts_and_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            profile = _profile("20251005_四01_ACT-4_云_J20_22#01", (10033,))
            partial_entry = PartialDataEntry(
                sortie_id="20251110_单01_ACT-2_涛_J20_26#01",
                source_type="self_built_inventory",
                stream_kind="vehicle",
                data_tier="Tier B",
                raw_manifest_path=None,
                inventory_reference="seed",
                time_range={"start_utc": None, "stop_utc": None},
                measurement_family=("BUS_PENDING",),
                usable_for_pretraining=True,
                notes=("vehicle-only",),
            )
            config = StageHExportConfig(
                run_id="stage-h-test",
                sortie_ids=(profile.sortie_id,),
                output_root=tmp / "artifacts" / "stage_h",
                report_path=tmp / "docs" / "reports" / "stage-h-test.md",
                partial_data_entries=(partial_entry,),
            )
            pipeline = StageHExportPipeline(
                config=config,
                profile_resolver=_FakeProfileResolver((profile,)),
                view_runner=_FakeViewRunner(include_stage_g=True),
                partial_data_builder=PartialDataBuilder(config=PartialDataConfig()),
            )

            result = pipeline.run()

            run_manifest_path = Path(result.run_manifest_path)
            report_path = Path(result.report_path)
            self.assertTrue(run_manifest_path.exists())
            self.assertTrue(report_path.exists())

            run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
            sortie_manifest_path = Path(run_manifest["sortie_manifest_paths"][profile.sortie_id])
            self.assertTrue(sortie_manifest_path.exists())
            sortie_manifest = json.loads(sortie_manifest_path.read_text(encoding="utf-8"))
            view_manifest_path = Path(sortie_manifest["view_manifest_paths"][profile.views[0].view_id])
            self.assertTrue(view_manifest_path.exists())
            view_manifest = json.loads(view_manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(view_manifest["view_id"], profile.views[0].view_id)
            self.assertEqual(run_manifest["generated_view_count"], 1)
            self.assertIn("partial_data", run_manifest)
            self.assertEqual(run_manifest["partial_data"]["entry_count"], 1)

            feature_bundle = np.load(view_manifest["artifact_paths"]["feature_bundle_npz"])
            self.assertEqual(
                set(feature_bundle.files),
                {
                    "physiology_reference_projection",
                    "vehicle_reference_projection",
                    "fused_representation",
                    "reference_offsets_s",
                    "attention_weights",
                    "vehicle_event_scores",
                },
            )
            self.assertTrue(Path(view_manifest["artifact_paths"]["projection_diagnostics_summary_json"]).exists())
            self.assertTrue(Path(view_manifest["artifact_paths"]["causal_fusion_summary_json"]).exists())
            self.assertIn("run_manifest.json", report_path.read_text(encoding="utf-8"))

    def test_dual_pilot_sortie_exports_two_views(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            profile = _profile("20251002_单01_ACT-8_翼云_J16_12#01", (10035, 10033))
            config = StageHExportConfig(
                run_id="stage-h-dual",
                sortie_ids=(profile.sortie_id,),
                output_root=tmp / "artifacts" / "stage_h",
                report_path=tmp / "docs" / "reports" / "stage-h-dual.md",
            )
            pipeline = StageHExportPipeline(
                config=config,
                profile_resolver=_FakeProfileResolver((profile,)),
                view_runner=_FakeViewRunner(include_stage_g=True),
            )

            result = pipeline.run()

            self.assertEqual(len(result.generated_view_ids), 2)
            for view_id in result.generated_view_ids:
                self.assertIn(view_id, {view.view_id for view in profile.views})

    def test_feature_bundle_uses_fixed_keys_without_stage_g(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            profile = _profile("20251005_四01_ACT-4_云_J20_22#01", (10033,))
            config = StageHExportConfig(
                run_id="stage-h-no-g",
                sortie_ids=(profile.sortie_id,),
                output_root=tmp / "artifacts" / "stage_h",
                report_path=tmp / "docs" / "reports" / "stage-h-no-g.md",
                causal_fusion_enabled=False,
            )
            pipeline = StageHExportPipeline(
                config=config,
                profile_resolver=_FakeProfileResolver((profile,)),
                view_runner=_FakeViewRunner(include_stage_g=False),
            )

            result = pipeline.run()
            run_manifest = json.loads(Path(result.run_manifest_path).read_text(encoding="utf-8"))
            sortie_manifest = json.loads(
                Path(run_manifest["sortie_manifest_paths"][profile.sortie_id]).read_text(encoding="utf-8")
            )
            view_manifest = json.loads(
                Path(sortie_manifest["view_manifest_paths"][profile.views[0].view_id]).read_text(encoding="utf-8")
            )
            feature_bundle = np.load(view_manifest["artifact_paths"]["feature_bundle_npz"])

            self.assertEqual(
                set(feature_bundle.files),
                {
                    "physiology_reference_projection",
                    "vehicle_reference_projection",
                    "fused_representation",
                    "reference_offsets_s",
                    "attention_weights",
                    "vehicle_event_scores",
                },
            )
            self.assertFalse(view_manifest["stage_g_available"])
            self.assertEqual(view_manifest["artifact_paths"]["causal_fusion_summary_json"], "")


class PartialDataBuilderTest(unittest.TestCase):
    def test_vehicle_only_partial_entry_builds_single_stream_artifacts(self) -> None:
        entry = PartialDataEntry(
            sortie_id="20251110_单01_ACT-2_涛_J20_26#01",
            source_type="self_built_inventory",
            stream_kind="vehicle",
            data_tier="Tier B",
            raw_manifest_path=None,
            inventory_reference="seed",
            time_range={"start_utc": "2025-11-10T01:00:00Z", "stop_utc": "2025-11-10T01:00:10Z"},
            measurement_family=("BUS9000",),
            usable_for_pretraining=True,
            notes=("vehicle-only",),
        )

        def point_provider(_: PartialDataEntry) -> tuple[RawPoint, ...]:
            return (
                RawPoint(StreamKind.VEHICLE, "BUS9000", _utc(2025, 11, 10, 1, 0, 0), {"speed": 200.0}),
                RawPoint(StreamKind.VEHICLE, "BUS9000", _utc(2025, 11, 10, 1, 0, 5), {"speed": 201.0}),
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            result = PartialDataBuilder(
                config=PartialDataConfig(),
                point_provider=point_provider,
            ).run((entry,), output_root=tmpdir)

            self.assertTrue(Path(result.manifest_path).exists())
            self.assertTrue(Path(result.window_manifest_path).exists())
            self.assertTrue(Path(result.feature_bundle_path or "").exists())
            self.assertGreaterEqual(len(result.built_samples), 1)
            rows = [
                json.loads(line)
                for line in Path(result.manifest_path).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(rows[0]["stream_kind"], "vehicle")
            self.assertEqual(rows[0]["builder_status"], "built")
