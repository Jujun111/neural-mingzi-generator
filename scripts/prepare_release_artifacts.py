import shutil
from pathlib import Path


ARTIFACTS = [
    ("lstm_weights.pth", "lstm_weights.pth"),
    ("vocab.pkl", "vocab.pkl"),
    ("transformer_weights.pth", "transformer_weights.pth"),
    ("transformer_vocab.pkl", "transformer_vocab.pkl"),
    ("markov_model.pkl", "markov_model.pkl"),
    ("artifacts/fim_template/best.ckpt", "fim_template_best.ckpt"),
    ("artifacts/fim_template/vocab.pkl", "fim_template_vocab.pkl"),
    ("artifacts/fim_template/config.json", "fim_template_config.json"),
]


def main() -> int:
    output_dir = Path("release_artifacts")
    output_dir.mkdir(parents=True, exist_ok=True)

    missing = [source for source, _ in ARTIFACTS if not Path(source).exists()]
    if missing:
        for source in missing:
            print(f"missing: {source}")
        return 1

    for source, target in ARTIFACTS:
        destination = output_dir / target
        shutil.copy2(source, destination)
        print(f"prepared: {destination}")

    print("Upload the files in release_artifacts/ to the GitHub Release.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
