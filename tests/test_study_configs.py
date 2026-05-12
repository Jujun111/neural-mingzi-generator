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

    def test_default_prompts_include_expected_compound_surnames(self):
        self.assertIn("欧阳", DEFAULT_PROMPTS)
        self.assertIn("司马", DEFAULT_PROMPTS)


if __name__ == "__main__":
    unittest.main()
