import json
import shutil
import unittest
from pathlib import Path

from study import resolve_parameter_count
from study_configs import count_trainable_parameters, get_model_spec


class FakeVocab:
    def __init__(self, size: int):
        self.size = size

    def __len__(self) -> int:
        return self.size


class StudyPipelineTests(unittest.TestCase):
    def test_resolve_parameter_count_prefers_training_config(self):
        output_dir = Path("tests") / "_tmp_resolve_parameter_count"
        output_dir.mkdir(parents=True, exist_ok=True)
        config_path = output_dir / "config.json"
        self.addCleanup(lambda: shutil.rmtree(output_dir, ignore_errors=True))
        config_path.write_text(
            json.dumps({"parameter_count": 123456}, ensure_ascii=False),
            encoding="utf-8",
        )

        resolved = resolve_parameter_count(
            model_name="lstm_small_matched",
            checkpoint_payload={"vocab": FakeVocab(9999)},
            output_dir=output_dir,
        )

        self.assertEqual(resolved, 123456)

    def test_resolve_parameter_count_falls_back_to_vocab_aware_count(self):
        vocab_size = 4321
        resolved = resolve_parameter_count(
            model_name="transformer_small",
            checkpoint_payload={"vocab": FakeVocab(vocab_size)},
            output_dir=Path("tests") / "_missing_resolve_parameter_count",
        )

        expected = count_trainable_parameters(get_model_spec("transformer_small"), vocab_size=vocab_size)
        self.assertEqual(resolved, expected)


if __name__ == "__main__":
    unittest.main()
