import unittest

from study_configs import (
    DEFAULT_PROMPTS,
    PRIMARY_MATCHED_PAIR,
    ROBUSTNESS_MATCHED_PAIR,
    count_trainable_parameters,
    get_model_spec,
)


class StudyConfigTests(unittest.TestCase):
    def test_primary_pair_is_parameter_matched_within_ten_percent(self):
        left = count_trainable_parameters(get_model_spec(PRIMARY_MATCHED_PAIR[0]))
        right = count_trainable_parameters(get_model_spec(PRIMARY_MATCHED_PAIR[1]))
        diff_ratio = abs(left - right) / max(left, right)
        self.assertLessEqual(diff_ratio, 0.10)

    def test_robustness_pair_is_parameter_matched_within_ten_percent(self):
        left = count_trainable_parameters(get_model_spec(ROBUSTNESS_MATCHED_PAIR[0]))
        right = count_trainable_parameters(get_model_spec(ROBUSTNESS_MATCHED_PAIR[1]))
        diff_ratio = abs(left - right) / max(left, right)
        self.assertLessEqual(diff_ratio, 0.10)

    def test_tuned_transformer_keeps_small_transformer_architecture(self):
        baseline = get_model_spec("transformer_small")
        tuned = get_model_spec("transformer_small_tuned")

        self.assertEqual(baseline.family, tuned.family)
        self.assertEqual(baseline.embedding_dim, tuned.embedding_dim)
        self.assertEqual(baseline.hidden_dim, tuned.hidden_dim)
        self.assertEqual(baseline.num_heads, tuned.num_heads)
        self.assertEqual(baseline.num_layers, tuned.num_layers)
        self.assertEqual(
            count_trainable_parameters(baseline),
            count_trainable_parameters(tuned),
        )

    def test_default_prompts_include_expected_compound_surnames(self):
        self.assertIn("欧阳", DEFAULT_PROMPTS)
        self.assertIn("司马", DEFAULT_PROMPTS)


if __name__ == "__main__":
    unittest.main()
