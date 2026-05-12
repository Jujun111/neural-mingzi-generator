import matplotlib.pyplot as plt
import numpy as np


# These values reconstruct the observed trend for illustration only.
# The middle LSTM epochs were not preserved in the original training logs.
lstm_known_start = [4.5268, 4.3971, 4.3557, 4.3287, 4.3084]
lstm_known_end = [4.1682, 4.1654, 4.1622, 4.1594, 4.1575, 4.1555]
lstm_middle = np.linspace(lstm_known_start[-1], lstm_known_end[0], 21)[1:-1].tolist()
lstm_loss_30 = lstm_known_start + lstm_middle + lstm_known_end

tf_loss_30 = [
    4.6049,
    4.5057,
    4.4740,
    4.4552,
    4.4426,
    4.4330,
    4.4261,
    4.4199,
    4.4148,
    4.4093,
    4.4059,
    4.4039,
    4.4001,
    4.3974,
    4.3952,
    4.3926,
    4.3910,
    4.3891,
    4.3876,
    4.3858,
    4.3844,
    4.3817,
    4.3827,
    4.3810,
    4.3804,
    4.3797,
    4.3793,
    4.3781,
    4.3767,
    4.3755,
]


def plot_real_curves(lstm_loss, tf_loss, save_path="real_convergence_plot.png"):
    lstm_epochs = range(1, len(lstm_loss) + 1)
    tf_epochs = range(1, len(tf_loss) + 1)

    lstm_ppl = [np.exp(loss) for loss in lstm_loss]
    tf_ppl = [np.exp(loss) for loss in tf_loss]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(lstm_epochs, lstm_loss, label="LSTM (reconstructed)", marker="o", color="blue")
    ax1.plot(tf_epochs, tf_loss, label="Transformer (logged)", marker="s", color="orange")
    ax1.set_title("Illustrative Convergence Comparison")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend()

    ax2.plot(lstm_epochs, lstm_ppl, label="LSTM PPL", marker="o", color="blue")
    ax2.plot(tf_epochs, tf_ppl, label="Transformer PPL", marker="s", color="orange")
    ax2.set_title("Illustrative Perplexity Comparison")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Perplexity")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)


if __name__ == "__main__":
    plot_real_curves(lstm_loss_30, tf_loss_30)
