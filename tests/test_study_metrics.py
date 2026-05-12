import unittest

from study_metrics import canonicalize_compound_prefix, summarize_generation_metrics


class StudyMetricsTests(unittest.TestCase):
    def test_generation_metrics_include_expected_fields(self):
        names = ["李白", "王安石", "欧阳修", "白白"]
        training_names = ["李白", "杜甫"]
        prompted_records = [
            {"prompt": "欧阳", "name": "欧阳修"},
            {"prompt": "司马", "name": "司马光"},
            {"prompt": "李", "name": "李白"},
        ]

        metrics = summarize_generation_metrics(names, training_names, prompted_records=prompted_records)

        self.assertIn("unique_name_ratio", metrics)
        self.assertIn("novelty_ratio", metrics)
        self.assertIn("train_overlap_ratio", metrics)
        self.assertIn("compound_surname_compatibility", metrics)
        self.assertGreaterEqual(metrics["compound_surname_compatibility"], 0.0)

    def test_compound_prompt_compatibility_handles_traditional_variants(self):
        prompted_records = [
            {"prompt": "欧阳", "name": "歐陽修"},
            {"prompt": "司马", "name": "司馬光"},
            {"prompt": "欧阳", "name": "歐陽白"},
            {"prompt": "李", "name": "李白"},
        ]

        metrics = summarize_generation_metrics(
            ["歐陽修", "司馬光", "歐陽白"],
            training_names=["李白"],
            prompted_records=prompted_records,
        )

        self.assertEqual(canonicalize_compound_prefix("歐陽"), "欧阳")
        self.assertEqual(canonicalize_compound_prefix("司馬"), "司马")
        self.assertEqual(metrics["compound_surname_compatibility"], 1.0)


if __name__ == "__main__":
    unittest.main()
