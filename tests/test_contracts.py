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
classifier_contract = load_module(
    "classifier_contract", "services/classifier_contract.py"
)
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


class ClassifierMetadataContractTest(unittest.TestCase):
    class FakeModel:
        model_name = "hgb"
        label_scheme = "9class"
        class_names = list(stroke_labels.CLASS_NAMES_9)
        classes_ = [0, 2, 3, 4, 5, 7]

    def test_pro_artifact_reports_six_trained_labels_inside_nine_class_schema(self):
        contract = classifier_contract.describe_classifier(
            "pro", self.FakeModel(), vit_available=True
        )

        self.assertEqual("9class", contract["feature_scheme"])
        self.assertEqual(6, contract["trained_label_count"])
        self.assertEqual(
            ["Serve", "Lob", "Smash", "Drop", "Drive", "Clear"],
            contract["trained_labels"],
        )
        self.assertIn(
            "scheme=9class trained_labels=6[Serve,Lob,Smash,Drop,Drive,Clear]",
            classifier_contract.format_classifier_contract(contract),
        )

    def test_callback_schemes_include_feature_and_vit_results(self):
        schemes = classifier_contract.collect_stroke_class_schemes([
            {"stroke_source": "feature_classifier", "stroke_class_scheme": "4class_amateur"},
            {"stroke_source": "vit_only"},
            {"stroke_source": None},
        ])

        self.assertEqual(["4class_amateur", "9class"], schemes)


if __name__ == "__main__":
    unittest.main()
