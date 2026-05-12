from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

from markov_chain import ChineseNameMarkov
from model import ChineseNameLSTM, ChineseNameTransformer


@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str
    size_tier: str
    embedding_dim: Optional[int] = None
    hidden_size: Optional[int] = None
    hidden_dim: Optional[int] = None
    num_heads: Optional[int] = None
    num_layers: Optional[int] = None
    max_seq_len: int = 10
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def build_model(self, vocab_size: int, pad_idx: int, dropout_prob: float):
        if self.family == "lstm":
            return ChineseNameLSTM(
                vocab_size=vocab_size,
                embedding_dim=self.embedding_dim or 256,
                hidden_size=self.hidden_size or 512,
                num_layers=self.num_layers or 2,
                pad_idx=pad_idx,
                dropout_prob=dropout_prob,
            )
        if self.family == "transformer":
            return ChineseNameTransformer(
                vocab_size=vocab_size,
                embedding_dim=self.embedding_dim or 256,
                num_heads=self.num_heads or 8,
                hidden_dim=self.hidden_dim or 512,
                num_layers=self.num_layers or 2,
                max_seq_len=self.max_seq_len,
                pad_idx=pad_idx,
                dropout_prob=dropout_prob,
            )
        raise ValueError(f"Model family '{self.family}' is not a neural model.")


MODEL_SPECS: Dict[str, ModelSpec] = {
    "markov_baseline": ModelSpec(
        name="markov_baseline",
        family="markov",
        size_tier="baseline",
        description="First-order probabilistic baseline.",
    ),
    "transformer_small": ModelSpec(
        name="transformer_small",
        family="transformer",
        size_tier="small_matched",
        embedding_dim=256,
        hidden_dim=512,
        num_heads=8,
        num_layers=2,
        description="Existing small causal Transformer used as the primary small matched reference.",
    ),
    "lstm_small_matched": ModelSpec(
        name="lstm_small_matched",
        family="lstm",
        size_tier="small_matched",
        embedding_dim=160,
        hidden_size=320,
        num_layers=2,
        description="Smaller LSTM matched to the small Transformer within a tight parameter budget.",
    ),
    "lstm_large": ModelSpec(
        name="lstm_large",
        family="lstm",
        size_tier="large_matched",
        embedding_dim=256,
        hidden_size=512,
        num_layers=2,
        description="Existing larger LSTM used as the large matched reference.",
    ),
    "transformer_large_matched": ModelSpec(
        name="transformer_large_matched",
        family="transformer",
        size_tier="large_matched",
        embedding_dim=384,
        hidden_dim=768,
        num_heads=8,
        num_layers=3,
        description="Larger Transformer matched to the existing larger LSTM.",
    ),
}

PRIMARY_MATCHED_PAIR = ("lstm_small_matched", "transformer_small")
ROBUSTNESS_MATCHED_PAIR = ("lstm_large", "transformer_large_matched")
DEFAULT_STUDY_MODELS = (
    "markov_baseline",
    "lstm_small_matched",
    "transformer_small",
    "lstm_large",
    "transformer_large_matched",
)
DEFAULT_STUDY_SEEDS = (13, 37, 73)
DEFAULT_DATA_FRACTIONS = (0.1, 0.3, 1.0)
DEFAULT_TEMPERATURES = (0.6, 0.8, 1.0)
DEFAULT_PROMPTS = (
    "\u674e",
    "\u738b",
    "\u5f20",
    "\u6b27\u9633",
    "\u53f8\u9a6c",
)


def get_model_spec(model_name: str) -> ModelSpec:
    if model_name not in MODEL_SPECS:
        raise KeyError(f"Unknown study model '{model_name}'.")
    return MODEL_SPECS[model_name]


def count_trainable_parameters(model_spec: ModelSpec, vocab_size: int = 8864) -> int:
    if model_spec.family == "markov":
        return 0
    model = model_spec.build_model(vocab_size=vocab_size, pad_idx=0, dropout_prob=0.0)
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def build_markov_model(surnames, boost_compound: bool = False) -> ChineseNameMarkov:
    return ChineseNameMarkov(surnames_list=surnames, boost_compound=boost_compound)
