import hashlib
import json
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from dataset import ChineseNameDataset, Vocabulary
from db_loader import get_cbdb_names


DEFAULT_SPLIT_SEED = 20260419
DEFAULT_FRACTION_SEED = 20260419


def load_name_pairs(
    db_path: str = "latest.db",
    dynasty_id: int | None = None,
    limit: int | None = None,
    sample_data_path: str | None = None,
) -> List[Tuple[str, str]]:
    if sample_data_path:
        payload = json.loads(Path(sample_data_path).read_text(encoding="utf-8"))
        surnames = payload.get("surnames", [])
        given_names = payload.get("given_names", [])
    else:
        surnames, given_names = get_cbdb_names(db_path=db_path, dynasty_id=dynasty_id, limit=limit)

    pairs = [(surname, given_name) for surname, given_name in zip(surnames, given_names) if surname and given_name]
    return sorted(pairs, key=lambda pair: (pair[0] + pair[1], pair[0], pair[1]))


def pairs_to_full_names(pairs: Sequence[Tuple[str, str]]) -> List[str]:
    return [surname + given_name for surname, given_name in pairs]


def compute_dataset_signature(full_names: Sequence[str]) -> str:
    digest = hashlib.sha1()
    for name in full_names:
        digest.update(name.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def create_fixed_split(
    full_names: Sequence[str],
    output_path: str | Path,
    split_seed: int = DEFAULT_SPLIT_SEED,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    split_name: str = "cbdb_fixed_v1",
) -> Dict[str, object]:
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-8:
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.0")

    total_count = len(full_names)
    indices = list(range(total_count))
    random.Random(split_seed).shuffle(indices)

    train_end = int(total_count * train_ratio)
    val_end = train_end + int(total_count * val_ratio)

    manifest = {
        "split_name": split_name,
        "split_seed": split_seed,
        "dataset_size": total_count,
        "dataset_signature": compute_dataset_signature(full_names),
        "ratios": {"train": train_ratio, "val": val_ratio, "test": test_ratio},
        "indices": {
            "train": indices[:train_end],
            "val": indices[train_end:val_end],
            "test": indices[val_end:],
        },
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def load_split_manifest(split_path: str | Path) -> Dict[str, object]:
    return json.loads(Path(split_path).read_text(encoding="utf-8"))


def verify_split_manifest(full_names: Sequence[str], manifest: Dict[str, object]) -> None:
    current_signature = compute_dataset_signature(full_names)
    if current_signature != manifest["dataset_signature"]:
        raise ValueError(
            "Dataset signature does not match the split manifest. Regenerate the split or restore the source data."
        )


def materialize_split(
    full_names: Sequence[str], manifest: Dict[str, object]
) -> Dict[str, List[str]]:
    verify_split_manifest(full_names, manifest)
    return {
        split_name: [full_names[index] for index in indices]
        for split_name, indices in manifest["indices"].items()
    }


def materialize_pair_split(
    pairs: Sequence[Tuple[str, str]], manifest: Dict[str, object]
) -> Dict[str, List[Tuple[str, str]]]:
    full_names = pairs_to_full_names(pairs)
    verify_split_manifest(full_names, manifest)
    return {
        split_name: [pairs[index] for index in indices]
        for split_name, indices in manifest["indices"].items()
    }


def select_fraction_subset(
    items: Sequence,
    fraction: float,
    fraction_seed: int = DEFAULT_FRACTION_SEED,
) -> List:
    if fraction <= 0.0 or fraction > 1.0:
        raise ValueError("fraction must be in (0.0, 1.0].")

    if fraction == 1.0:
        return list(items)

    subset_size = max(1, int(len(items) * fraction))
    indices = list(range(len(items)))
    random.Random(fraction_seed).shuffle(indices)
    chosen_indices = sorted(indices[:subset_size])
    return [items[index] for index in chosen_indices]


def build_shared_vocabulary(train_full_names: Sequence[str], frequency_threshold: int = 1) -> Vocabulary:
    vocab = Vocabulary(frequency_threshold=frequency_threshold)
    vocab.build_vocabulary(list(train_full_names))
    return vocab


def build_name_dataset(full_names: Sequence[str], vocab: Vocabulary, max_seq_len: int = 6) -> ChineseNameDataset:
    return ChineseNameDataset(list(full_names), vocab, max_seq_len=max_seq_len)


def fraction_tag(fraction: float) -> str:
    return f"fraction_{int(round(fraction * 100)):03d}"
