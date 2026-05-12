import unittest
from pathlib import Path
import shutil

from study_data import create_fixed_split, load_name_pairs, load_split_manifest, materialize_split, pairs_to_full_names, select_fraction_subset


class StudyDataTests(unittest.TestCase):
    def test_split_manifest_is_reproducible(self):
        repo_root = Path(__file__).resolve().parents[1]
        pairs = load_name_pairs(sample_data_path=str(repo_root / "data" / "sample_names.json"))
        full_names = pairs_to_full_names(pairs)
        tmp_dir = repo_root / "data" / "_test_split_artifacts_a"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        try:
            split_path = tmp_dir / "split.json"
            create_fixed_split(full_names, split_path, split_seed=1234, split_name="sample")
            manifest_a = load_split_manifest(split_path)
            create_fixed_split(full_names, split_path, split_seed=1234, split_name="sample")
            manifest_b = load_split_manifest(split_path)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        self.assertEqual(manifest_a["indices"], manifest_b["indices"])

    def test_materialized_split_covers_dataset(self):
        repo_root = Path(__file__).resolve().parents[1]
        pairs = load_name_pairs(sample_data_path=str(repo_root / "data" / "sample_names.json"))
        full_names = pairs_to_full_names(pairs)
        tmp_dir = repo_root / "data" / "_test_split_artifacts_b"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        try:
            split_path = tmp_dir / "split.json"
            create_fixed_split(full_names, split_path, split_seed=2024, split_name="sample")
            manifest = load_split_manifest(split_path)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        split_names = materialize_split(full_names, manifest)
        total = len(split_names["train"]) + len(split_names["val"]) + len(split_names["test"])
        self.assertEqual(total, len(full_names))

    def test_fraction_subset_is_deterministic(self):
        items = list(range(100))
        subset_a = select_fraction_subset(items, fraction=0.3, fraction_seed=99)
        subset_b = select_fraction_subset(items, fraction=0.3, fraction_seed=99)
        self.assertEqual(subset_a, subset_b)


if __name__ == "__main__":
    unittest.main()
