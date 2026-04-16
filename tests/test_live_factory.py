"""Tests for Stage B live loader factory wiring."""

from __future__ import annotations

import sys
from pathlib import Path
import unittest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chronaris.access.live_factory import StageBLiveLoaderConfig


class LiveFactoryConfigTest(unittest.TestCase):
    def test_stage_b_live_loader_config_defaults_are_sane(self) -> None:
        config = StageBLiveLoaderConfig(
            java_properties_path="human-machine.properties",
            access_rule_id=6000019510066,
            analysis_id=6000019110020,
        )

        self.assertEqual(config.mysql_database, "rjgx_backend")
        self.assertIsNone(config.physiology_measurements)
        self.assertIsNone(config.physiology_point_limit_per_measurement)
        self.assertIsNone(config.bus_point_limit)
        self.assertIsNone(config.start_time_override_utc)
        self.assertIsNone(config.stop_time_override_utc)

    def test_stage_b_live_loader_config_accepts_time_overrides(self) -> None:
        config = StageBLiveLoaderConfig(
            java_properties_path="human-machine.properties",
            access_rule_id=6000019510066,
            analysis_id=6000019110020,
            start_time_override_utc=datetime(2025, 10, 5, 1, 35, 0, tzinfo=timezone.utc),
            stop_time_override_utc=datetime(2025, 10, 5, 1, 38, 1, tzinfo=timezone.utc),
        )

        self.assertEqual(config.start_time_override_utc.isoformat(), "2025-10-05T01:35:00+00:00")
        self.assertEqual(config.stop_time_override_utc.isoformat(), "2025-10-05T01:38:01+00:00")


if __name__ == "__main__":
    unittest.main()
from datetime import datetime, timezone
