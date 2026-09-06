from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {relative_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


analysis_mode = load_module("analysis_mode_contract", "services/analysis_mode.py")
stroke_labels = load_module("stroke_labels_contract", "analysis/stroke_labels.py")


class AnalysisModeContractTest(unittest.TestCase):
    def test_pro_profile_favors_throughput(self):
        mode, profile = analysis_mode.get_analysis_mode_profile("pro")

        self.assertEqual("pro", mode)
        self.assertEqual(4, profile["tracknet_batch_size"])
        self.assertFalse(profile["run_near_miss_rescue"])

    def test_amateur_profile_favors_recall(self):
        mode, profile = analysis_mode.get_analysis_mode_profile("amateur")

        self.assertEqual("amateur", mode)
        self.assertEqual(1, profile["tracknet_batch_size"])
        self.assertTrue(profile["run_near_miss_rescue"])

    def test_unknown_mode_uses_safe_pro_default(self):
        self.assertEqual(
            "pro", analysis_mode.get_analysis_mode_profile("unexpected")[0]
        )
        self.assertEqual("pro", analysis_mode.get_analysis_mode_profile(None)[0])

    def test_profile_mutation_does_not_change_global_contract(self):
        _, profile = analysis_mode.get_analysis_mode_profile("pro")
        profile["tracknet_batch_size"] = 999

        _, next_profile = analysis_mode.get_analysis_mode_profile("pro")
        self.assertEqual(4, next_profile["tracknet_batch_size"])


class StrokeLabelContractTest(unittest.TestCase):
    def test_pro_classes_and_mapping(self):
        self.assertEqual(6, len(stroke_labels.CLASS_NAMES_6_PRO))
        self.assertEqual("Clear", stroke_labels.map_9_to_6_pro("Lob"))
        self.assertEqual("Drive", stroke_labels.map_9_to_6_pro("Push"))
        self.assertIsNone(stroke_labels.map_9_to_6_pro("Defense"))

    def test_amateur_classes_and_mapping(self):
        self.assertEqual(4, len(stroke_labels.CLASS_NAMES_4_AMATEUR))
        self.assertEqual("Clear", stroke_labels.map_9_to_4_amateur("Lob"))
        self.assertEqual("Clear", stroke_labels.map_9_to_4_amateur("Net"))
        self.assertEqual("Drive", stroke_labels.map_9_to_4_amateur("Drop"))

    def test_unknown_label_is_rejected(self):
        with self.assertRaises(KeyError):
            stroke_labels.map_9_to_6_pro("Unknown")


if __name__ == "__main__":
    unittest.main()
