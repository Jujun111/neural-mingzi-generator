from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def plot_learning_curves(
    lstm_loss: List[float], tf_loss: List[float], save_path: str = "convergence_plot.png"
) -> None:
    """Plot cross-entropy loss and perplexity for both neural models."""
    epochs = range(1, len(lstm_loss) + 1)
    lstm_ppl = [np.exp(loss) for loss in lstm_loss]
    tf_ppl = [np.exp(loss) for loss in tf_loss]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, lstm_loss, label="LSTM Loss", marker="o", color="blue")
    ax1.plot(epochs, tf_loss, label="Transformer Loss", marker="s", color="orange")
    ax1.set_title("Training Convergence: Cross-Entropy Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend()

    ax2.plot(epochs, lstm_ppl, label="LSTM PPL", marker="o", color="blue")
    ax2.plot(epochs, tf_ppl, label="Transformer PPL", marker="s", color="orange")
    ax2.set_title("Model Uncertainty: Perplexity (Lower is Better)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Perplexity")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)


def evaluate_generation_diversity(generated_names: List[str]) -> Dict[str, float]:
    """
    Compute light-weight lexical diversity metrics for generated samples.
    """
    total_names = len(generated_names)
    if total_names == 0:
        return {}

    unique_ratio = len(set(generated_names)) / total_names
    lengths = [len(name) for name in generated_names]
    avg_length = sum(lengths) / total_names

    names_with_any_duplicates = sum(1 for name in generated_names if len(set(name)) < len(name))
    names_with_consecutive_duplicates = sum(
        1
        for name in generated_names
        if any(name[index] == name[index + 1] for index in range(len(name) - 1))
    )

    metrics = {
        "Total Samples Generated": total_names,
        "Unique Name Ratio": round(unique_ratio, 4),
        "Average Name Length": round(avg_length, 2),
        "Names with Any Duplicate Chars Ratio": round(names_with_any_duplicates / total_names, 4),
        "Names with Consecutive Duplicate Chars Ratio": round(
            names_with_consecutive_duplicates / total_names, 4
        ),
    }

    print("\n--- Generation Diversity Metrics ---")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    return metrics


if __name__ == "__main__":
    mock_lstm_loss = [4.5, 4.3, 4.2, 4.18, 4.16]
    mock_tf_loss = [4.6, 4.5, 4.46, 4.41, 4.38]
    plot_learning_curves(mock_lstm_loss, mock_tf_loss)

    mock_generations = ["李白", "杜甫", "欧阳修", "李白", "白白", "王安石", "苏轼", "欧阳修"]
    evaluate_generation_diversity(mock_generations)
