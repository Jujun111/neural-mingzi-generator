import argparse
import csv
import json
import math
import pickle
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import opencc
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from inference import (
    build_fim_prompt,
    generate_fim_name,
    generate_name,
    token_is_single_occurrence,
    token_matches_position,
)
from model import ChineseNameTransformer
from study_configs import (
    DEFAULT_DATA_FRACTIONS,
    DEFAULT_PROMPTS,
    DEFAULT_STUDY_MODELS,
    DEFAULT_STUDY_SEEDS,
    DEFAULT_TEMPERATURES,
    PRIMARY_MATCHED_PAIR,
    ROBUSTNESS_MATCHED_PAIR,
    build_markov_model,
    count_trainable_parameters,
    get_model_spec,
)
from study_data import (
    DEFAULT_FRACTION_SEED,
    DEFAULT_SPLIT_SEED,
    build_name_dataset,
    build_shared_vocabulary,
    create_fixed_split,
    fraction_tag,
    load_name_pairs,
    load_split_manifest,
    materialize_pair_split,
    materialize_split,
    pairs_to_full_names,
    select_fraction_subset,
)
from study_metrics import summarize_generation_metrics, teacher_forced_metrics


WEB_EVAL_CURATED_DYNASTIES = (
    {"dynasty_id": 6, "code": "Tang", "label_en": "Tang", "label_zh": "唐"},
    {"dynasty_id": 15, "code": "Song", "label_en": "Song", "label_zh": "宋"},
    {"dynasty_id": 19, "code": "Ming", "label_en": "Ming", "label_zh": "明"},
    {"dynasty_id": 20, "code": "Qing", "label_en": "Qing", "label_zh": "清"},
)


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str) -> str:
    if device_name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available in this environment.")
    return device_name


def parse_csv_floats(value: str) -> List[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_csv_ints(value: str) -> List[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_csv_strings(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def json_dump(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run_directory(
    output_root: Path,
    run_id: str,
    model_name: str,
    seed: int,
    split_name: str,
    fraction: float,
) -> Path:
    return output_root / run_id / model_name / f"seed_{seed}" / f"split_{split_name}" / fraction_tag(fraction)


def load_study_dataset(args) -> tuple[list[tuple[str, str]], list[str], dict]:
    pairs = load_name_pairs(
        db_path=args.db_path,
        dynasty_id=args.dynasty,
        limit=args.limit,
        sample_data_path=args.sample_data_path,
    )
    full_names = pairs_to_full_names(pairs)
    manifest = load_split_manifest(args.split_path)
    return pairs, full_names, manifest


def resolve_parameter_count(
    model_name: str,
    checkpoint_payload: Dict[str, object],
    output_dir: Path,
) -> int:
    if model_name == "markov_baseline":
        return 0

    config_path = output_dir / "config.json"
    if config_path.exists():
        config_payload = json.loads(config_path.read_text(encoding="utf-8"))
        if "parameter_count" in config_payload:
            return int(config_payload["parameter_count"])

    vocab = checkpoint_payload.get("vocab")
    if vocab is not None:
        return count_trainable_parameters(get_model_spec(model_name), vocab_size=len(vocab))

    return count_trainable_parameters(get_model_spec(model_name))


def save_history_csv(path: Path, history_rows: Sequence[Dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["epoch", "train_loss", "train_perplexity", "val_loss", "val_perplexity", "is_best"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history_rows)


def prepare_dataloaders(
    split_full_names: Dict[str, List[str]],
    fraction: float,
    batch_size: int,
    max_seq_len: int,
):
    full_train_names = list(split_full_names["train"])
    train_names = select_fraction_subset(full_train_names, fraction=fraction, fraction_seed=DEFAULT_FRACTION_SEED)
    val_names = list(split_full_names["val"])
    test_names = list(split_full_names["test"])
    vocab = build_shared_vocabulary(full_train_names)

    datasets = {
        "train": build_name_dataset(train_names, vocab, max_seq_len=max_seq_len),
        "val": build_name_dataset(val_names, vocab, max_seq_len=max_seq_len),
        "test": build_name_dataset(test_names, vocab, max_seq_len=max_seq_len),
    }
    dataloaders = {
        split_name: DataLoader(dataset, batch_size=batch_size, shuffle=(split_name == "train"), drop_last=False)
        for split_name, dataset in datasets.items()
    }
    return vocab, train_names, val_names, test_names, dataloaders


def run_neural_training(
    spec_name: str,
    output_dir: Path,
    vocab,
    dataloaders,
    device: str,
    seed: int,
    fraction: float,
    split_name: str,
    learning_rate: float,
    max_epochs: int,
    patience: int,
    dropout_prob: float,
) -> Dict[str, object]:
    spec = get_model_spec(spec_name)
    set_random_seed(seed)
    model = spec.build_model(vocab_size=len(vocab), pad_idx=vocab.PAD_IDX, dropout_prob=dropout_prob).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=vocab.PAD_IDX, reduction="sum")

    history_rows = []
    best_val_ppl = float("inf")
    best_epoch = 0
    best_train_loss = float("inf")
    epochs_without_improvement = 0

    config_payload = {
        "run_type": "neural_training",
        "model_spec": spec.to_dict(),
        "parameter_count": count_trainable_parameters(spec, vocab_size=len(vocab)),
        "seed": seed,
        "fraction": fraction,
        "split_name": split_name,
        "learning_rate": learning_rate,
        "max_epochs": max_epochs,
        "patience": patience,
        "dropout_prob": dropout_prob,
    }
    json_dump(output_dir / "config.json", config_payload)

    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss = 0.0
        total_tokens = 0

        for x_batch, y_batch in dataloaders["train"]:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()

            logits, _ = model(x_batch)
            logits_flat = logits.view(-1, logits.size(-1))
            y_flat = y_batch.view(-1)
            loss = criterion(logits_flat, y_flat)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += float(loss.item())
            total_tokens += int((y_flat != vocab.PAD_IDX).sum().item())

        train_loss = total_loss / max(total_tokens, 1)
        val_stats = teacher_forced_metrics(model, dataloaders["val"], pad_idx=vocab.PAD_IDX, device=device)
        is_best = val_stats["perplexity"] < best_val_ppl
        if is_best:
            best_val_ppl = val_stats["perplexity"]
            best_epoch = epoch
            best_train_loss = train_loss
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_spec": spec.to_dict(),
                    "state_dict": model.state_dict(),
                    "vocab": vocab,
                    "seed": seed,
                    "fraction": fraction,
                    "split_name": split_name,
                    "best_epoch": epoch,
                },
                output_dir / "best.ckpt",
            )
        else:
            epochs_without_improvement += 1

        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 8),
                "train_perplexity": round(math.exp(train_loss), 8),
                "val_loss": round(val_stats["cross_entropy"], 8),
                "val_perplexity": round(val_stats["perplexity"], 8),
                "is_best": int(is_best),
            }
        )

        if epochs_without_improvement >= patience:
            break

    torch.save(
        {
            "model_spec": spec.to_dict(),
            "state_dict": model.state_dict(),
            "vocab": vocab,
            "seed": seed,
            "fraction": fraction,
            "split_name": split_name,
            "best_epoch": best_epoch,
        },
        output_dir / "final.ckpt",
    )

    save_history_csv(output_dir / "history.csv", history_rows)
    metrics_payload = {
        "family": spec.family,
        "parameter_count": count_trainable_parameters(spec, vocab_size=len(vocab)),
        "best_epoch": best_epoch,
        "best_val_perplexity": round(best_val_ppl, 8),
        "best_train_loss": round(best_train_loss, 8),
        "train_val_gap": round(best_val_ppl - math.exp(best_train_loss), 8),
        "epochs_completed": len(history_rows),
    }
    json_dump(output_dir / "metrics.json", metrics_payload)
    return metrics_payload


def run_markov_training(
    output_dir: Path,
    train_pairs: Sequence[tuple[str, str]],
    seed: int,
    fraction: float,
    split_name: str,
) -> Dict[str, object]:
    surnames = [surname for surname, _ in train_pairs]
    given_names = [given_name for _, given_name in train_pairs]
    model = build_markov_model(surnames=surnames, boost_compound=False)
    model.train(given_names)

    config_payload = {
        "run_type": "markov_training",
        "model_spec": get_model_spec("markov_baseline").to_dict(),
        "seed": seed,
        "fraction": fraction,
        "split_name": split_name,
    }
    json_dump(output_dir / "config.json", config_payload)

    checkpoint_payload = {
        "model_spec": get_model_spec("markov_baseline").to_dict(),
        "model": model,
        "seed": seed,
        "fraction": fraction,
        "split_name": split_name,
    }
    torch.save(checkpoint_payload, output_dir / "best.ckpt")
    torch.save(checkpoint_payload, output_dir / "final.ckpt")
    save_history_csv(
        output_dir / "history.csv",
        [
            {
                "epoch": 0,
                "train_loss": 0.0,
                "train_perplexity": 0.0,
                "val_loss": 0.0,
                "val_perplexity": 0.0,
                "is_best": 1,
            }
        ],
    )

    metrics_payload = {
        "family": "markov",
        "parameter_count": 0,
        "best_epoch": 0,
        "best_val_perplexity": None,
        "best_train_loss": None,
        "train_val_gap": None,
        "epochs_completed": 0,
    }
    json_dump(output_dir / "metrics.json", metrics_payload)
    return metrics_payload


def generate_samples(
    model_family: str,
    checkpoint_payload: Dict[str, object],
    device: str,
    sample_count: int,
    temperature: float,
    generation_seed: int,
    prompts: Sequence[str],
    prompted_sample_count: int,
) -> List[Dict[str, object]]:
    set_random_seed(generation_seed)
    records: List[Dict[str, object]] = []

    if model_family == "markov":
        model = checkpoint_payload["model"]
        for index in range(sample_count):
            name = model.generate(max_length=2)
            records.append(
                {
                    "sample_type": "unprompted",
                    "prompt": None,
                    "index": index,
                    "temperature": None,
                    "name": name,
                }
            )

        per_prompt = max(1, prompted_sample_count // max(len(prompts), 1))
        for prompt in prompts:
            for index in range(per_prompt):
                name = model.generate(max_length=2, seed_surname=prompt)
                records.append(
                    {
                        "sample_type": "prompted",
                        "prompt": prompt,
                        "index": index,
                        "temperature": None,
                        "name": name,
                    }
                )
        return records

    model_spec = checkpoint_payload["model_spec"]
    spec_name = model_spec["name"]
    vocab = checkpoint_payload["vocab"]
    spec = get_model_spec(spec_name)
    model = spec.build_model(vocab_size=len(vocab), pad_idx=vocab.PAD_IDX, dropout_prob=0.0).to(device)
    model.load_state_dict(checkpoint_payload["state_dict"])
    model.eval()

    for index in range(sample_count):
        name = generate_name(
            model=model,
            vocab=vocab,
            seed_text=None,
            max_length=4,
            temperature=temperature,
            device=device,
        )
        records.append(
            {
                "sample_type": "unprompted",
                "prompt": None,
                "index": index,
                "temperature": temperature,
                "name": name,
            }
        )

    per_prompt = max(1, prompted_sample_count // max(len(prompts), 1))
    for prompt in prompts:
        for index in range(per_prompt):
            name = generate_name(
                model=model,
                vocab=vocab,
                seed_text=prompt,
                max_length=4,
                temperature=temperature,
                device=device,
            )
            records.append(
                {
                    "sample_type": "prompted",
                    "prompt": prompt,
                    "index": index,
                    "temperature": temperature,
                    "name": name,
                }
            )
    return records


def write_samples_jsonl(path: Path, records: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_samples_jsonl(path: Path) -> List[Dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def train_command(args) -> None:
    pairs, full_names, manifest = load_study_dataset(args)
    split_pairs = materialize_pair_split(pairs, manifest)
    split_full_names = materialize_split(full_names, manifest)
    split_name = manifest["split_name"]

    output_dir = run_directory(
        output_root=Path(args.output_root),
        run_id=args.run_id,
        model_name=args.model_spec,
        seed=args.seed,
        split_name=split_name,
        fraction=args.fraction,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.model_spec == "markov_baseline":
        train_pairs = select_fraction_subset(
            split_pairs["train"], fraction=args.fraction, fraction_seed=DEFAULT_FRACTION_SEED
        )
        run_markov_training(
            output_dir=output_dir,
            train_pairs=train_pairs,
            seed=args.seed,
            fraction=args.fraction,
            split_name=split_name,
        )
        print(f"Markov baseline artifacts written to {output_dir}")
        return

    vocab, _, _, _, dataloaders = prepare_dataloaders(
        split_full_names=split_full_names,
        fraction=args.fraction,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
    )
    metrics = run_neural_training(
        spec_name=args.model_spec,
        output_dir=output_dir,
        vocab=vocab,
        dataloaders=dataloaders,
        device=resolve_device(args.device),
        seed=args.seed,
        fraction=args.fraction,
        split_name=split_name,
        learning_rate=args.learning_rate,
        max_epochs=args.max_epochs,
        patience=args.patience,
        dropout_prob=args.dropout_prob,
    )
    print(f"Training complete for {args.model_spec}. Best val perplexity: {metrics['best_val_perplexity']}")


def fim_position_for_span(name: str, start: int, span_length: int) -> str:
    if start == 0:
        return "start"
    if start + span_length >= len(name):
        return "end"
    return "middle"


def build_fim_template_records(
    full_names: Sequence[str],
    seed: int,
    max_spans_per_name: int = 1,
    include_any_position: bool = True,
) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    rng = random.Random(seed)
    for name_index, name in enumerate(full_names):
        spans: List[tuple[int, int]] = []
        for span_length in (1, 2):
            if len(name) < span_length:
                continue
            for start in range(0, len(name) - span_length + 1):
                spans.append((start, span_length))

        rng.shuffle(spans)
        for start, span_length in spans[:max_spans_per_name]:
            fixed_token = name[start : start + span_length]
            position = fim_position_for_span(name, start, span_length)
            optional_seed = name[:1]
            positions = [position]
            if include_any_position and position != "any":
                positions.append("any")
            for prompt_position in positions:
                prompt = (
                    f"<TASK_INFILL> <TOKEN> {fixed_token} <POSITION> {prompt_position} "
                    f"<SEED> {optional_seed} <SEP>"
                )
                records.append(
                    {
                        "record_id": f"fim-{name_index}-{start}-{span_length}-{prompt_position}",
                        "fixed_token": fixed_token,
                        "position": prompt_position,
                        "seed": optional_seed,
                        "prompt": prompt,
                        "target": name,
                        "loss_mask": "target_after_sep",
                    }
                )
    return records


class FIMTemplateDataset(Dataset):
    def __init__(self, records: Sequence[Dict[str, object]], vocab, max_seq_len: int):
        self.vocab = vocab
        self.max_seq_len = max_seq_len
        self.X: List[List[int]] = []
        self.Y: List[List[int]] = []
        self.loss_mask: List[List[int]] = []

        for record in records:
            prompt = str(record["prompt"])
            target = str(record["target"])
            prompt_indices = vocab.encode(prompt)
            target_indices = vocab.encode(target)
            sequence = [vocab.SOS_IDX, *prompt_indices, *target_indices, vocab.EOS_IDX]
            supervised_start = 1 + len(prompt_indices)

            if len(sequence) > max_seq_len:
                sequence = sequence[:max_seq_len]
                sequence[-1] = vocab.EOS_IDX

            x_seq = sequence[:-1]
            y_seq = sequence[1:]
            mask_seq = [1 if index >= supervised_start - 1 else 0 for index in range(len(y_seq))]
            pad_len = max_seq_len - 1 - len(x_seq)
            if pad_len < 0:
                continue

            self.X.append(x_seq + [vocab.PAD_IDX] * pad_len)
            self.Y.append(y_seq + [vocab.PAD_IDX] * pad_len)
            self.loss_mask.append(mask_seq + [0] * pad_len)

        self.X_tensor = torch.tensor(self.X, dtype=torch.long)
        self.Y_tensor = torch.tensor(self.Y, dtype=torch.long)
        self.mask_tensor = torch.tensor(self.loss_mask, dtype=torch.bool)

    def __len__(self) -> int:
        return len(self.X_tensor)

    def __getitem__(self, index: int):
        return self.X_tensor[index], self.Y_tensor[index], self.mask_tensor[index]


def build_fim_vocabulary(train_records: Sequence[Dict[str, object]]):
    corpus = [str(record["prompt"]) + str(record["target"]) for record in train_records]
    return build_shared_vocabulary(corpus)


def fim_allowed_output_indices(vocab) -> List[int]:
    return [
        index
        for char, index in vocab.char2idx.items()
        if len(char) == 1 and "\u4e00" <= char <= "\u9fff"
    ]


def run_fim_epoch(model, dataloader, criterion, optimizer, device: str, train: bool) -> Dict[str, float]:
    model.train(mode=train)
    total_loss = 0.0
    total_tokens = 0

    for x_batch, y_batch, mask_batch in dataloader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        mask_batch = mask_batch.to(device)
        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            logits, _ = model(x_batch)
            y_masked = y_batch.clone()
            y_masked[~mask_batch] = -100
            loss = criterion(logits.view(-1, logits.size(-1)), y_masked.view(-1))
            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

        token_count = int(mask_batch.sum().item())
        total_loss += float(loss.item())
        total_tokens += token_count

    cross_entropy = total_loss / max(total_tokens, 1)
    return {
        "cross_entropy": cross_entropy,
        "perplexity": math.exp(min(cross_entropy, 20)),
        "tokens": total_tokens,
    }


def train_fim_template_command(args) -> None:
    _, full_names, manifest = load_study_dataset(args)
    split_full_names = materialize_split(full_names, manifest)
    split_name = manifest["split_name"]
    train_names = select_fraction_subset(
        split_full_names["train"],
        fraction=args.fraction,
        fraction_seed=DEFAULT_FRACTION_SEED,
    )
    val_names = list(split_full_names["val"])
    output_dir = run_directory(
        output_root=Path(args.output_root),
        run_id=args.run_id,
        model_name="fim_template",
        seed=args.seed,
        split_name=split_name,
        fraction=args.fraction,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    set_random_seed(args.seed)
    train_records = build_fim_template_records(
        train_names,
        seed=args.seed,
        max_spans_per_name=args.max_spans_per_name,
        include_any_position=not args.no_any_position_records,
    )
    val_records = build_fim_template_records(
        val_names,
        seed=args.seed + 1,
        max_spans_per_name=args.max_spans_per_name,
        include_any_position=not args.no_any_position_records,
    )
    vocab = build_fim_vocabulary(train_records)
    with (output_dir / "vocab.pkl").open("wb") as handle:
        pickle.dump(vocab, handle)

    write_samples_jsonl(output_dir / "fim_train.jsonl", train_records)
    write_samples_jsonl(output_dir / "fim_val.jsonl", val_records)
    train_dataset = FIMTemplateDataset(train_records, vocab, max_seq_len=args.fim_max_seq_len)
    val_dataset = FIMTemplateDataset(val_records, vocab, max_seq_len=args.fim_max_seq_len)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    model = ChineseNameTransformer(
        vocab_size=len(vocab),
        embedding_dim=args.embedding_dim,
        num_heads=args.num_heads,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        max_seq_len=args.fim_max_seq_len,
        pad_idx=vocab.PAD_IDX,
        dropout_prob=args.dropout_prob,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="sum")
    allowed_indices = fim_allowed_output_indices(vocab)

    config_payload = {
        "run_type": "fim_template_training",
        "model_family": "conditional_decoder_only_transformer",
        "status": "trained",
        "seed": args.seed,
        "fraction": args.fraction,
        "split_name": split_name,
        "prompt_format": "<TASK_INFILL> <TOKEN> fixed_token <POSITION> position <SEED> optional_seed <SEP>",
        "loss_policy": "loss_only_after_sep_target_region",
        "embedding_dim": args.embedding_dim,
        "num_heads": args.num_heads,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "max_seq_len": args.fim_max_seq_len,
        "max_new_tokens": args.max_new_tokens,
        "dropout_prob": args.dropout_prob,
        "learning_rate": args.learning_rate,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "max_spans_per_name": args.max_spans_per_name,
        "include_any_position_records": not args.no_any_position_records,
    }
    json_dump(output_dir / "config.json", config_payload)

    history_rows = []
    best_val_ppl = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    best_train_loss = float("inf")
    for epoch in range(1, args.max_epochs + 1):
        train_stats = run_fim_epoch(model, train_loader, criterion, optimizer, device=device, train=True)
        val_stats = run_fim_epoch(model, val_loader, criterion, optimizer=None, device=device, train=False)
        is_best = val_stats["perplexity"] < best_val_ppl
        if is_best:
            best_val_ppl = val_stats["perplexity"]
            best_epoch = epoch
            best_train_loss = train_stats["cross_entropy"]
            epochs_without_improvement = 0
            torch.save(
                {
                    "artifact_type": "fim_template_decoder",
                    "config": config_payload,
                    "state_dict": model.state_dict(),
                    "vocab": vocab,
                    "allowed_output_indices": allowed_indices,
                    "seed": args.seed,
                    "fraction": args.fraction,
                    "split_name": split_name,
                    "best_epoch": epoch,
                },
                output_dir / "best.ckpt",
            )
        else:
            epochs_without_improvement += 1

        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": round(train_stats["cross_entropy"], 8),
                "train_perplexity": round(train_stats["perplexity"], 8),
                "val_loss": round(val_stats["cross_entropy"], 8),
                "val_perplexity": round(val_stats["perplexity"], 8),
                "is_best": int(is_best),
            }
        )
        print(
            "FIM epoch "
            f"{epoch}: train_ppl={train_stats['perplexity']:.4f}, "
            f"val_ppl={val_stats['perplexity']:.4f}, "
            f"best={int(is_best)}"
        )
        if epochs_without_improvement >= args.patience:
            break

    torch.save(
        {
            "artifact_type": "fim_template_decoder",
            "config": config_payload,
            "state_dict": model.state_dict(),
            "vocab": vocab,
            "allowed_output_indices": allowed_indices,
            "seed": args.seed,
            "fraction": args.fraction,
            "split_name": split_name,
            "best_epoch": best_epoch,
        },
        output_dir / "final.ckpt",
    )
    save_history_csv(output_dir / "history.csv", history_rows)
    metrics_payload = {
        "family": "fim_template",
        "training_status": "trained",
        "train_name_count": len(train_names),
        "val_name_count": len(val_names),
        "train_record_count": len(train_records),
        "val_record_count": len(val_records),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "best_epoch": best_epoch,
        "best_val_perplexity": round(best_val_ppl, 8),
        "best_train_loss": round(best_train_loss, 8),
        "constraint_satisfaction_rate": None,
    }
    json_dump(output_dir / "metrics.json", metrics_payload)
    print(f"FIM template decoder checkpoint written to {output_dir}")


def evaluate_fim_template_command(args) -> None:
    _, full_names, manifest = load_study_dataset(args)
    split_full_names = materialize_split(full_names, manifest)
    split_name = manifest["split_name"]
    test_names = list(split_full_names["test"])
    output_dir = run_directory(
        output_root=Path(args.output_root),
        run_id=args.run_id,
        model_name="fim_template",
        seed=args.seed,
        split_name=split_name,
        fraction=args.fraction,
    )
    checkpoint_path = output_dir / f"{args.checkpoint}.ckpt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"FIM template checkpoint not found at {checkpoint_path}")

    converter = opencc.OpenCC("s2t")
    fixed_tokens = [converter.convert(token) for token in parse_csv_strings(args.fixed_tokens)]
    checkpoint_payload = torch.load(checkpoint_path, map_location=resolve_device(args.device), weights_only=False)
    config = checkpoint_payload["config"]
    vocab = checkpoint_payload["vocab"]
    model = ChineseNameTransformer(
        vocab_size=len(vocab),
        embedding_dim=int(config["embedding_dim"]),
        num_heads=int(config["num_heads"]),
        hidden_dim=int(config["hidden_dim"]),
        num_layers=int(config["num_layers"]),
        max_seq_len=int(config["max_seq_len"]),
        pad_idx=vocab.PAD_IDX,
        dropout_prob=0.0,
    ).to(resolve_device(args.device))
    model.load_state_dict(checkpoint_payload["state_dict"])
    model.eval()
    allowed_indices = checkpoint_payload.get("allowed_output_indices") or fim_allowed_output_indices(vocab)
    rows: List[Dict[str, object]] = []
    satisfied = 0
    total = 0
    for fixed_token in fixed_tokens:
        for index in range(args.sample_count):
            name = ""
            is_satisfied = False
            attempts = 0
            for attempts in range(1, args.attempts_per_sample + 1):
                name = generate_fim_name(
                    model=model,
                    vocab=vocab,
                    fixed_token=fixed_token,
                    position=args.position,
                seed_text=None,
                max_new_tokens=args.max_new_tokens,
                min_new_tokens=args.min_new_tokens,
                temperature=args.temperature,
                    device=resolve_device(args.device),
                    allowed_output_indices=allowed_indices,
                )
                is_satisfied = token_matches_position(name, fixed_token, args.position)
                if is_satisfied and token_is_single_occurrence(name, fixed_token):
                    break
            total += 1
            single_occurrence = token_is_single_occurrence(name, fixed_token)
            satisfied += int(is_satisfied and single_occurrence)
            rows.append(
                {
                    "fixed_token": fixed_token,
                    "position": args.position,
                    "index": index,
                    "temperature": args.temperature,
                    "name": name,
                    "satisfies_constraint": is_satisfied,
                    "single_fixed_token_occurrence": single_occurrence,
                    "sampling_attempts": attempts,
                    "source": "fim_template_decoder",
                }
            )

    write_samples_jsonl(output_dir / "fim_eval_samples.jsonl", rows)
    unique_names = {str(row["name"]) for row in rows}
    train_names = set(split_full_names["train"])
    metrics_payload = {
        "family": "fim_template",
        "evaluation_status": "decoder_generation",
        "fixed_tokens": fixed_tokens,
        "position": args.position,
        "temperature": args.temperature,
        "sample_count": len(rows),
        "attempts_per_sample": args.attempts_per_sample,
        "min_new_tokens": args.min_new_tokens,
        "max_new_tokens": args.max_new_tokens,
        "constraint_satisfaction_rate": round(satisfied / max(total, 1), 8),
        "unique_name_ratio": round(len(unique_names) / max(len(rows), 1), 8),
        "duplicate_ratio": round(1 - (len(unique_names) / max(len(rows), 1)), 8),
        "novelty_ratio": round(sum(1 for name in unique_names if name not in train_names) / max(len(unique_names), 1), 8),
    }
    json_dump(output_dir / "test_metrics.json", metrics_payload)
    print(f"FIM template evaluation written to {output_dir}")


def evaluate_command(args) -> None:
    pairs, full_names, manifest = load_study_dataset(args)
    split_full_names = materialize_split(full_names, manifest)
    split_name = manifest["split_name"]
    output_dir = run_directory(
        output_root=Path(args.output_root),
        run_id=args.run_id,
        model_name=args.model_spec,
        seed=args.seed,
        split_name=split_name,
        fraction=args.fraction,
    )
    checkpoint_path = output_dir / ("best.ckpt" if args.checkpoint == "best" else "final.ckpt")
    checkpoint_payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    device = resolve_device(args.device)

    eval_dir = output_dir / "eval" / f"temperature_{args.temperature:.1f}"
    eval_dir.mkdir(parents=True, exist_ok=True)

    if args.model_spec == "markov_baseline":
        teacher_forced = {"val": None, "test": None}
        train_names = pairs_to_full_names(
            select_fraction_subset(
                materialize_pair_split(pairs, manifest)["train"],
                fraction=args.fraction,
                fraction_seed=DEFAULT_FRACTION_SEED,
            )
        )
    else:
        vocab = checkpoint_payload["vocab"]
        _, train_names, _, test_names, dataloaders = prepare_dataloaders(
            split_full_names=split_full_names,
            fraction=args.fraction,
            batch_size=args.batch_size,
            max_seq_len=args.max_seq_len,
        )
        spec = get_model_spec(args.model_spec)
        model = spec.build_model(vocab_size=len(vocab), pad_idx=vocab.PAD_IDX, dropout_prob=0.0).to(device)
        model.load_state_dict(checkpoint_payload["state_dict"])
        teacher_forced = {
            "val": teacher_forced_metrics(model, dataloaders["val"], pad_idx=vocab.PAD_IDX, device=device),
            "test": teacher_forced_metrics(model, dataloaders["test"], pad_idx=vocab.PAD_IDX, device=device),
        }

    if args.model_spec == "markov_baseline":
        test_names = list(split_full_names["test"])

    records = generate_samples(
        model_family=get_model_spec(args.model_spec).family,
        checkpoint_payload=checkpoint_payload,
        device=device,
        sample_count=args.sample_count,
        temperature=args.temperature,
        generation_seed=args.seed + int(args.temperature * 1000),
        prompts=parse_csv_strings(args.prompts),
        prompted_sample_count=args.prompted_sample_count,
    )
    write_samples_jsonl(eval_dir / "samples.jsonl", records)

    unprompted_names = [record["name"] for record in records if record["sample_type"] == "unprompted"]
    prompted_records = [record for record in records if record["sample_type"] == "prompted"]

    metrics_payload = {
        "run_id": args.run_id,
        "model_spec": args.model_spec,
        "model_family": get_model_spec(args.model_spec).family,
        "size_tier": get_model_spec(args.model_spec).size_tier,
        "seed": args.seed,
        "fraction": args.fraction,
        "split_name": split_name,
        "temperature": args.temperature,
        "checkpoint": args.checkpoint,
        "teacher_forced": teacher_forced,
        "generation": {
            "unprompted": summarize_generation_metrics(unprompted_names, training_names=train_names),
            "prompted": summarize_generation_metrics(
                [record["name"] for record in prompted_records],
                training_names=train_names,
                prompted_records=prompted_records,
            ),
        },
        "parameter_count": resolve_parameter_count(
            model_name=args.model_spec,
            checkpoint_payload=checkpoint_payload,
            output_dir=output_dir,
        ),
    }
    json_dump(eval_dir / "test_metrics.json", metrics_payload)
    print(f"Evaluation artifacts written to {eval_dir}")


def collect_metric_rows(output_root: Path, run_id: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    base_dir = output_root / run_id
    if not base_dir.exists():
        return rows

    for metrics_path in base_dir.glob("**/eval/temperature_*/test_metrics.json"):
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        row = {
            "run_id": payload["run_id"],
            "model_spec": payload["model_spec"],
            "model_family": payload["model_family"],
            "size_tier": payload["size_tier"],
            "seed": payload["seed"],
            "fraction": payload["fraction"],
            "temperature": payload["temperature"],
            "parameter_count": payload["parameter_count"],
            "test_perplexity": None
            if payload["teacher_forced"]["test"] is None
            else payload["teacher_forced"]["test"]["perplexity"],
            "val_perplexity": None
            if payload["teacher_forced"]["val"] is None
            else payload["teacher_forced"]["val"]["perplexity"],
            "unique_name_ratio": payload["generation"]["unprompted"]["unique_name_ratio"],
            "distinct_1": payload["generation"]["unprompted"]["distinct_1"],
            "distinct_2": payload["generation"]["unprompted"]["distinct_2"],
            "novelty_ratio": payload["generation"]["unprompted"]["novelty_ratio"],
            "train_overlap_ratio": payload["generation"]["unprompted"]["train_overlap_ratio"],
            "consecutive_duplicate_ratio": payload["generation"]["unprompted"]["consecutive_duplicate_ratio"],
            "any_duplicate_ratio": payload["generation"]["unprompted"]["any_duplicate_ratio"],
            "invalid_output_ratio": payload["generation"]["unprompted"]["invalid_output_ratio"],
            "compound_surname_compatibility": payload["generation"]["prompted"].get(
                "compound_surname_compatibility"
            ),
        }
        rows.append(row)

    rows.sort(key=lambda row: (row["model_spec"], row["fraction"], row["seed"], row["temperature"]))
    return rows


def summarize_command(args) -> None:
    rows = collect_metric_rows(output_root=Path(args.output_root), run_id=args.run_id)
    summary_dir = Path(args.output_root) / args.run_id / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "run_id",
        "model_spec",
        "model_family",
        "size_tier",
        "seed",
        "fraction",
        "temperature",
        "parameter_count",
        "test_perplexity",
        "val_perplexity",
        "unique_name_ratio",
        "distinct_1",
        "distinct_2",
        "novelty_ratio",
        "train_overlap_ratio",
        "consecutive_duplicate_ratio",
        "any_duplicate_ratio",
        "invalid_output_ratio",
        "compound_surname_compatibility",
    ]
    with (summary_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    json_dump(summary_dir / "summary.json", {"rows": rows})
    print(f"Summary written to {summary_dir}")


def export_human_eval_command(args) -> None:
    output_root = Path(args.output_root)
    run_id = args.run_id
    model_names = ["markov_baseline", *PRIMARY_MATCHED_PAIR]
    split_name = load_split_manifest(args.split_path)["split_name"]

    rows = []
    key_rows = []
    blinded_index = 1

    for model_name in model_names:
        candidate_dir = None
        candidate_metrics = None
        best_score = float("inf")

        for seed in parse_csv_ints(args.seeds):
            run_dir = run_directory(
                output_root=output_root,
                run_id=run_id,
                model_name=model_name,
                seed=seed,
                split_name=split_name,
                fraction=1.0,
            )
            metrics_path = run_dir / "eval" / f"temperature_{args.temperature:.1f}" / "test_metrics.json"
            if not metrics_path.exists():
                continue
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            score = payload["teacher_forced"]["val"]["perplexity"] if payload["teacher_forced"]["val"] else 0.0
            if candidate_dir is None or score < best_score:
                candidate_dir = run_dir
                candidate_metrics = payload
                best_score = score

        if candidate_dir is None:
            raise FileNotFoundError(f"No evaluated run found for {model_name} at temperature {args.temperature}.")

        samples_path = candidate_dir / "eval" / f"temperature_{args.temperature:.1f}" / "samples.jsonl"
        sample_records = load_samples_jsonl(samples_path)
        unprompted = [record for record in sample_records if record["sample_type"] == "unprompted"][: args.unprompted_count]
        prompted = [record for record in sample_records if record["sample_type"] == "prompted"][: args.prompted_count]

        for record in [*unprompted, *prompted]:
            blind_id = f"C{blinded_index:03d}"
            blinded_index += 1
            rows.append(
                {
                    "candidate_id": blind_id,
                    "sample_type": record["sample_type"],
                    "prompt": record["prompt"] or "",
                    "name": record["name"],
                    "naturalness": "",
                    "realism": "",
                    "phonetic_flow": "",
                    "novelty_balance": "",
                    "notes": "",
                }
            )
            key_rows.append(
                {
                    "candidate_id": blind_id,
                    "model_spec": model_name,
                    "seed": candidate_metrics["seed"],
                    "temperature": args.temperature,
                    "source_run_dir": str(candidate_dir),
                    "sample_type": record["sample_type"],
                    "prompt": record["prompt"] or "",
                    "name": record["name"],
                }
            )

    random.Random(args.blind_seed).shuffle(rows)
    export_dir = output_root / run_id / "human_eval"
    export_dir.mkdir(parents=True, exist_ok=True)

    with (export_dir / "human_eval.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "candidate_id",
                "sample_type",
                "prompt",
                "name",
                "naturalness",
                "realism",
                "phonetic_flow",
                "novelty_balance",
                "notes",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    with (export_dir / "human_eval_key.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "candidate_id",
                "model_spec",
                "seed",
                "temperature",
                "source_run_dir",
                "sample_type",
                "prompt",
                "name",
            ],
        )
        writer.writeheader()
        writer.writerows(key_rows)

    print(f"Human evaluation sheets written to {export_dir}")


def _select_best_evaluated_run(
    *,
    output_root: Path,
    run_id: str,
    model_name: str,
    split_name: str,
    temperature: float,
    seeds: Sequence[int],
) -> tuple[Path, Dict[str, object]]:
    candidate_dir = None
    candidate_metrics = None
    best_score = float("inf")

    for seed in seeds:
        run_dir = run_directory(
            output_root=output_root,
            run_id=run_id,
            model_name=model_name,
            seed=seed,
            split_name=split_name,
            fraction=1.0,
        )
        metrics_path = run_dir / "eval" / f"temperature_{temperature:.1f}" / "test_metrics.json"
        if not metrics_path.exists():
            continue

        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        val_payload = payload.get("teacher_forced", {}).get("val")
        score = float(val_payload["perplexity"]) if val_payload else 0.0
        if candidate_dir is None or score < best_score:
            candidate_dir = run_dir
            candidate_metrics = payload
            best_score = score

    if candidate_dir is None or candidate_metrics is None:
        raise FileNotFoundError(
            f"No evaluated run found for {model_name} at temperature {temperature:.1f}."
        )

    return candidate_dir, candidate_metrics


def export_web_eval_tasks_command(args) -> None:
    output_root = Path(args.output_root)
    split_name = load_split_manifest(args.split_path)["split_name"]
    seeds = parse_csv_ints(args.seeds)
    randomizer = random.Random(args.blind_seed)

    lstm_dir, lstm_metrics = _select_best_evaluated_run(
        output_root=output_root,
        run_id=args.run_id,
        model_name=PRIMARY_MATCHED_PAIR[0],
        split_name=split_name,
        temperature=args.temperature,
        seeds=seeds,
    )
    transformer_dir, transformer_metrics = _select_best_evaluated_run(
        output_root=output_root,
        run_id=args.run_id,
        model_name=PRIMARY_MATCHED_PAIR[1],
        split_name=split_name,
        temperature=args.temperature,
        seeds=seeds,
    )

    lstm_samples = load_samples_jsonl(
        lstm_dir / "eval" / f"temperature_{args.temperature:.1f}" / "samples.jsonl"
    )
    transformer_samples = load_samples_jsonl(
        transformer_dir / "eval" / f"temperature_{args.temperature:.1f}" / "samples.jsonl"
    )

    sample_type = args.sample_type
    lstm_names = [record["name"] for record in lstm_samples if record["sample_type"] == sample_type]
    transformer_names = [
        record["name"] for record in transformer_samples if record["sample_type"] == sample_type
    ]
    task_count = min(args.task_count, len(lstm_names) // args.batch_size, len(transformer_names) // args.batch_size)
    if task_count <= 0:
        raise RuntimeError("Not enough exported generation samples to build the blind pairwise task pool.")

    blind_pairwise_tasks = []
    for index in range(task_count):
        start = index * args.batch_size
        left_names = lstm_names[start : start + args.batch_size]
        right_names = transformer_names[start : start + args.batch_size]
        if randomizer.random() < 0.5:
            left_names, right_names = right_names, left_names
            left_model, right_model = "transformer", "lstm"
        else:
            left_model, right_model = "lstm", "transformer"

        blind_pairwise_tasks.append(
            {
                "task_id": f"bp_{index + 1:03d}",
                "task_type": "blind_pairwise",
                "run_id": args.run_id,
                "checkpoint_id": f"{PRIMARY_MATCHED_PAIR[0]}__{PRIMARY_MATCHED_PAIR[1]}__temperature_{args.temperature:.1f}",
                "prompt_type": sample_type,
                "title": "Blind Pairwise",
                "prompt": "Which batch feels more like plausible Chinese names?",
                "presented_payload": {
                    "left": {"label": "A", "names": left_names},
                    "right": {"label": "B", "names": right_names},
                },
                "response_options": ["left", "right", "tie", "skip"],
                "hidden_answer": {
                    "left_model_type": left_model,
                    "right_model_type": right_model,
                    "lstm_seed": lstm_metrics["seed"],
                    "transformer_seed": transformer_metrics["seed"],
                },
            }
        )

    dynasty_guess_tasks = []
    for dynasty in WEB_EVAL_CURATED_DYNASTIES:
        dynasty_pairs = load_name_pairs(
            db_path=args.db_path,
            dynasty_id=dynasty["dynasty_id"],
            limit=args.dynasty_limit,
            sample_data_path=args.sample_data_path,
        )
        if not dynasty_pairs:
            raise RuntimeError(
                f"No dynasty-conditioned name pairs were available for {dynasty['code']} ({dynasty['dynasty_id']})."
            )

        surnames = [surname for surname, _ in dynasty_pairs]
        given_names = [given_name for _, given_name in dynasty_pairs]
        model = build_markov_model(surnames=surnames, boost_compound=False)
        model.train(given_names)

        for task_index in range(args.dynasty_tasks_per_dynasty):
            names = [model.generate(max_length=2) for _ in range(args.batch_size)]
            dynasty_guess_tasks.append(
                {
                    "task_id": f"dg_{len(dynasty_guess_tasks) + 1:03d}",
                    "task_type": "dynasty_guess",
                    "run_id": args.run_id,
                    "checkpoint_id": f"markov_dynasty_{dynasty['code'].lower()}",
                    "prompt_type": "dynasty_style",
                    "title": "Dynasty Guess",
                    "prompt": "Which dynasty does this batch feel closest to?",
                    "presented_payload": {"names": names},
                    "response_options": ["Tang", "Song", "Ming", "Qing", "skip"],
                    "hidden_answer": {
                        "dynasty_id": dynasty["dynasty_id"],
                        "dynasty_code": dynasty["code"],
                    },
                }
            )

    export_path = Path(args.output_path)
    export_path.parent.mkdir(parents=True, exist_ok=True)
    json_dump(
        export_path,
        {
            "version": "1.0",
            "run_id": args.run_id,
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "task_pools": {
                "blind_pairwise": blind_pairwise_tasks,
                "dynasty_guess": dynasty_guess_tasks,
            },
        },
    )
    print(f"Web evaluation task pool written to {export_path}")


def run_default_study_command(args) -> None:
    models_full_fraction = list(DEFAULT_STUDY_MODELS)
    fractions = parse_csv_floats(args.fractions)
    temperatures = parse_csv_floats(args.temperatures)
    seeds = parse_csv_ints(args.seeds)

    for model_name in models_full_fraction:
        for seed in seeds:
            train_args = argparse.Namespace(**vars(args))
            train_args.model_spec = model_name
            train_args.seed = seed
            train_args.fraction = 1.0
            train_command(train_args)
            for temperature in temperatures:
                eval_args = argparse.Namespace(**vars(args))
                eval_args.model_spec = model_name
                eval_args.seed = seed
                eval_args.fraction = 1.0
                eval_args.temperature = temperature
                eval_args.checkpoint = "best"
                evaluate_command(eval_args)

    for model_name in PRIMARY_MATCHED_PAIR:
        for fraction in fractions:
            if fraction == 1.0:
                continue
            for seed in seeds:
                train_args = argparse.Namespace(**vars(args))
                train_args.model_spec = model_name
                train_args.seed = seed
                train_args.fraction = fraction
                train_command(train_args)
                for temperature in temperatures:
                    eval_args = argparse.Namespace(**vars(args))
                    eval_args.model_spec = model_name
                    eval_args.seed = seed
                    eval_args.fraction = fraction
                    eval_args.temperature = temperature
                    eval_args.checkpoint = "best"
                    evaluate_command(eval_args)

    summarize_command(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Study pipeline for the Chinese Name Generator project.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    split_parser = subparsers.add_parser("prepare-split", help="Create a fixed train/val/test split manifest.")
    split_parser.add_argument("--db-path", default="latest.db")
    split_parser.add_argument("--sample-data-path", default=None)
    split_parser.add_argument("--dynasty", type=int, default=None)
    split_parser.add_argument("--limit", type=int, default=None)
    split_parser.add_argument("--split-path", default="data/cbdb_fixed_split_v1.json")
    split_parser.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    split_parser.add_argument("--split-name", default="cbdb_fixed_v1")
    split_parser.set_defaults(
        func=lambda args: create_fixed_split(
            pairs_to_full_names(
                load_name_pairs(
                    db_path=args.db_path,
                    dynasty_id=args.dynasty,
                    limit=args.limit,
                    sample_data_path=args.sample_data_path,
                )
            ),
            output_path=args.split_path,
            split_seed=args.split_seed,
            split_name=args.split_name,
        )
    )

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--db-path", default="latest.db")
    common.add_argument("--sample-data-path", default=None)
    common.add_argument("--dynasty", type=int, default=None)
    common.add_argument("--limit", type=int, default=None)
    common.add_argument("--split-path", default="data/cbdb_fixed_split_v1.json")
    common.add_argument("--run-id", default="fairness_v1")
    common.add_argument("--output-root", default="results")
    common.add_argument("--device", default="auto")
    common.add_argument("--batch-size", type=int, default=128)
    common.add_argument("--max-seq-len", type=int, default=6)

    train_parser = subparsers.add_parser("train", parents=[common], help="Train a single study run.")
    train_parser.add_argument("--model-spec", required=True, choices=list(DEFAULT_STUDY_MODELS))
    train_parser.add_argument("--seed", type=int, required=True)
    train_parser.add_argument("--fraction", type=float, default=1.0)
    train_parser.add_argument("--learning-rate", type=float, default=0.001)
    train_parser.add_argument("--max-epochs", type=int, default=40)
    train_parser.add_argument("--patience", type=int, default=5)
    train_parser.add_argument("--dropout-prob", type=float, default=0.4)
    train_parser.set_defaults(func=train_command)

    fim_train_parser = subparsers.add_parser(
        "train-fim-template",
        parents=[common],
        help="Train a template conditional decoder-only Transformer for fixed-token infill.",
    )
    fim_train_parser.add_argument("--seed", type=int, required=True)
    fim_train_parser.add_argument("--fraction", type=float, default=1.0)
    fim_train_parser.add_argument("--fim-max-seq-len", type=int, default=96)
    fim_train_parser.add_argument("--max-new-tokens", type=int, default=4)
    fim_train_parser.add_argument("--embedding-dim", type=int, default=128)
    fim_train_parser.add_argument("--num-heads", type=int, default=4)
    fim_train_parser.add_argument("--hidden-dim", type=int, default=256)
    fim_train_parser.add_argument("--num-layers", type=int, default=2)
    fim_train_parser.add_argument("--learning-rate", type=float, default=0.001)
    fim_train_parser.add_argument("--max-epochs", type=int, default=20)
    fim_train_parser.add_argument("--patience", type=int, default=4)
    fim_train_parser.add_argument("--dropout-prob", type=float, default=0.2)
    fim_train_parser.add_argument("--max-spans-per-name", type=int, default=1)
    fim_train_parser.add_argument("--no-any-position-records", action="store_true")
    fim_train_parser.set_defaults(func=train_fim_template_command)

    fim_eval_parser = subparsers.add_parser(
        "evaluate-fim-template",
        parents=[common],
        help="Evaluate a trained template FIM decoder checkpoint.",
    )
    fim_eval_parser.add_argument("--seed", type=int, required=True)
    fim_eval_parser.add_argument("--fraction", type=float, default=1.0)
    fim_eval_parser.add_argument("--fixed-tokens", default="飞,妃,若虚")
    fim_eval_parser.add_argument("--sample-count", type=int, default=20)
    fim_eval_parser.add_argument("--position", choices=["any", "start", "middle", "end"], default="any")
    fim_eval_parser.add_argument("--temperature", type=float, default=0.8)
    fim_eval_parser.add_argument("--max-new-tokens", type=int, default=4)
    fim_eval_parser.add_argument("--min-new-tokens", type=int, default=2)
    fim_eval_parser.add_argument("--attempts-per-sample", type=int, default=20)
    fim_eval_parser.add_argument("--checkpoint", choices=["best", "final"], default="best")
    fim_eval_parser.set_defaults(func=evaluate_fim_template_command)

    evaluate_parser = subparsers.add_parser("evaluate", parents=[common], help="Evaluate a trained study run.")
    evaluate_parser.add_argument("--model-spec", required=True, choices=list(DEFAULT_STUDY_MODELS))
    evaluate_parser.add_argument("--seed", type=int, required=True)
    evaluate_parser.add_argument("--fraction", type=float, default=1.0)
    evaluate_parser.add_argument("--temperature", type=float, default=0.8)
    evaluate_parser.add_argument("--sample-count", type=int, default=2000)
    evaluate_parser.add_argument("--prompted-sample-count", type=int, default=250)
    evaluate_parser.add_argument("--prompts", default=",".join(DEFAULT_PROMPTS))
    evaluate_parser.add_argument("--checkpoint", choices=["best", "final"], default="best")
    evaluate_parser.set_defaults(func=evaluate_command)

    summarize_parser = subparsers.add_parser("summarize", parents=[common], help="Collect test metrics into a summary table.")
    summarize_parser.set_defaults(func=summarize_command)

    human_eval_parser = subparsers.add_parser(
        "export-human-eval", parents=[common], help="Export anonymized CSVs for lightweight human evaluation."
    )
    human_eval_parser.add_argument("--temperature", type=float, default=0.8)
    human_eval_parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_STUDY_SEEDS))
    human_eval_parser.add_argument("--unprompted-count", type=int, default=50)
    human_eval_parser.add_argument("--prompted-count", type=int, default=20)
    human_eval_parser.add_argument("--blind-seed", type=int, default=20260419)
    human_eval_parser.set_defaults(func=export_human_eval_command)

    web_eval_parser = subparsers.add_parser(
        "export-web-eval",
        parents=[common],
        help="Export pre-generated blind pairwise and dynasty-guess task pools for the public web app.",
    )
    web_eval_parser.add_argument("--temperature", type=float, default=0.8)
    web_eval_parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_STUDY_SEEDS))
    web_eval_parser.add_argument("--sample-type", choices=["unprompted", "prompted"], default="unprompted")
    web_eval_parser.add_argument("--task-count", type=int, default=20)
    web_eval_parser.add_argument("--dynasty-tasks-per-dynasty", type=int, default=5)
    web_eval_parser.add_argument("--dynasty-limit", type=int, default=None)
    web_eval_parser.add_argument("--blind-seed", type=int, default=20260419)
    web_eval_parser.add_argument(
        "--output-path",
        default="data/generated_eval_tasks.json",
    )
    web_eval_parser.set_defaults(func=export_web_eval_tasks_command)

    default_study_parser = subparsers.add_parser(
        "run-default-study",
        parents=[common],
        help="Run the default fairness-first study sweep: full-fraction main runs plus the primary data-scale ablation.",
    )
    default_study_parser.add_argument("--fractions", default=",".join(str(fraction) for fraction in DEFAULT_DATA_FRACTIONS))
    default_study_parser.add_argument(
        "--temperatures", default=",".join(str(temperature) for temperature in DEFAULT_TEMPERATURES)
    )
    default_study_parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_STUDY_SEEDS))
    default_study_parser.add_argument("--sample-count", type=int, default=2000)
    default_study_parser.add_argument("--prompted-sample-count", type=int, default=250)
    default_study_parser.add_argument("--prompts", default=",".join(DEFAULT_PROMPTS))
    default_study_parser.add_argument("--checkpoint", choices=["best", "final"], default="best")
    default_study_parser.add_argument("--learning-rate", type=float, default=0.001)
    default_study_parser.add_argument("--max-epochs", type=int, default=40)
    default_study_parser.add_argument("--patience", type=int, default=5)
    default_study_parser.add_argument("--dropout-prob", type=float, default=0.4)
    default_study_parser.set_defaults(func=run_default_study_command)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    result = args.func(args)
    if result is not None and args.command == "prepare-split":
        print(f"Split manifest written to {args.split_path}")


if __name__ == "__main__":
    main()
