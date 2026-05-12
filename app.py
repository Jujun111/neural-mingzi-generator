import contextlib
import json
import logging
import os
import pickle
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, List, Literal, Optional, Tuple

# Windows can load multiple OpenMP runtimes when torch and numeric deps are
# imported together; setting this early avoids a hard crash during local boot.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import opencc
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator, model_validator

from db_loader import get_cbdb_names
from evaluation_tasks import EvaluationTaskPool, SUPPORTED_TASK_TYPES
from feedback_store import (
    BaseFeedbackStore,
    StoredFeedbackEvent,
    build_feedback_store,
)
from markov_chain import ChineseNameMarkov


logging.basicConfig(
    level=os.getenv("CNG_LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
LOGGER = logging.getLogger("chinese_name_generator.api")

SUPPORTED_MODELS = ("markov", "lstm", "transformer")
SUPPORTED_TOKEN_POSITIONS = ("any", "start", "middle", "end")
SUPPORTED_SURFACES = ("playground", "evaluation_lab")
SUPPORTED_FEEDBACK_TASK_TYPES = ("favorite_pick", *SUPPORTED_TASK_TYPES)
BLIND_PAIRWISE_CHOICES = ("left", "right", "tie", "skip")
DYNASTY_GUESS_CHOICES = ("Tang", "Song", "Ming", "Qing", "skip")
CURATED_DYNASTIES: Tuple[Dict[str, Any], ...] = (
    {"dynasty_id": 6, "code": "Tang", "label_en": "Tang", "label_zh": "唐"},
    {"dynasty_id": 15, "code": "Song", "label_en": "Song", "label_zh": "宋"},
    {"dynasty_id": 19, "code": "Ming", "label_en": "Ming", "label_zh": "明"},
    {"dynasty_id": 20, "code": "Qing", "label_en": "Qing", "label_zh": "清"},
)
CURATED_DYNASTY_IDS = tuple(item["dynasty_id"] for item in CURATED_DYNASTIES)
ConstraintPosition = Literal["any", "start", "middle", "end"]


def _parse_csv_env(env_name: str, default: List[str]) -> List[str]:
    raw_value = os.getenv(env_name)
    if raw_value is None:
        return list(default)

    values = [value.strip() for value in raw_value.split(",")]
    return [value for value in values if value]


def _parse_bool_env(env_name: str, default: bool) -> bool:
    raw_value = os.getenv(env_name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_int_env(env_name: str, default: int) -> int:
    raw_value = os.getenv(env_name)
    if raw_value is None:
        return default
    return int(raw_value)


def normalize_prompt_text(
    prompt_text: Optional[str], converter: Optional[opencc.OpenCC] = None
) -> Tuple[Optional[str], bool]:
    """Normalize simplified Chinese prompts into the training domain without importing torch."""
    if not prompt_text:
        return None, False

    active_converter = converter or opencc.OpenCC("s2t")
    normalized_text = active_converter.convert(prompt_text)
    return normalized_text, normalized_text != prompt_text


def normalize_seed_text(
    seed_text: Optional[str], converter: Optional[opencc.OpenCC] = None
) -> Tuple[Optional[str], bool]:
    return normalize_prompt_text(seed_text, converter=converter)


def token_matches_position(
    name: str,
    token: Optional[str],
    position: ConstraintPosition = "any",
) -> bool:
    if not token:
        return True

    if position == "any":
        return token in name

    if position == "start":
        return name.startswith(token)

    if position == "end":
        return name.endswith(token)

    if position == "middle":
        token_length = len(token)
        max_start = len(name) - token_length - 1
        if max_start < 1:
            return False
        return any(name[index : index + token_length] == token for index in range(1, max_start + 1))

    raise ValueError(f"Unsupported token position '{position}'.")


def token_is_single_occurrence(name: str, token: Optional[str]) -> bool:
    if not token:
        return True
    return name.count(token) == 1


def _load_local_env_file(path: str = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_local_env_file()


class InMemoryRateLimiter:
    def __init__(self, window_seconds: int = 60):
        self.window_seconds = window_seconds
        self._events: Dict[str, Deque[float]] = defaultdict(deque)

    def allow(self, key: str, limit: int) -> bool:
        now = time.monotonic()
        bucket = self._events[key]

        while bucket and now - bucket[0] > self.window_seconds:
            bucket.popleft()

        if len(bucket) >= limit:
            return False

        bucket.append(now)
        return True


@dataclass
class AppSettings:
    artifact_dir: Path = field(
        default_factory=lambda: Path(os.getenv("CNG_ARTIFACT_DIR", ".")).resolve()
    )
    sample_data_path: Path = field(
        default_factory=lambda: Path(os.getenv("CNG_SAMPLE_DATA", "data/sample_names.json")).resolve()
    )
    db_path: Path = field(default_factory=lambda: Path(os.getenv("CNG_DB_PATH", "latest.db")).resolve())
    markov_artifact_path: Path = field(
        default_factory=lambda: Path(os.getenv("CNG_MARKOV_ARTIFACT", "markov_model.pkl")).resolve()
    )
    lstm_weights_path: Path = field(
        default_factory=lambda: Path(os.getenv("CNG_LSTM_WEIGHTS", "lstm_weights.pth")).resolve()
    )
    lstm_vocab_path: Path = field(
        default_factory=lambda: Path(os.getenv("CNG_LSTM_VOCAB", "vocab.pkl")).resolve()
    )
    transformer_weights_path: Path = field(
        default_factory=lambda: Path(os.getenv("CNG_TRANSFORMER_WEIGHTS", "transformer_weights.pth")).resolve()
    )
    transformer_vocab_path: Path = field(
        default_factory=lambda: Path(
            os.getenv("CNG_TRANSFORMER_VOCAB", "transformer_vocab.pkl")
        ).resolve()
    )
    enable_fim_mode: bool = field(
        default_factory=lambda: _parse_bool_env("CNG_ENABLE_FIM_MODE", False)
    )
    fim_weights_path: Path = field(
        default_factory=lambda: Path(
            os.getenv("CNG_FIM_WEIGHTS", "artifacts/fim_template/best.ckpt")
        ).resolve()
    )
    fim_vocab_path: Path = field(
        default_factory=lambda: Path(
            os.getenv("CNG_FIM_VOCAB", "artifacts/fim_template/vocab.pkl")
        ).resolve()
    )
    fim_config_path: Path = field(
        default_factory=lambda: Path(
            os.getenv("CNG_FIM_CONFIG", "artifacts/fim_template/config.json")
        ).resolve()
    )
    eval_task_pool_path: Path = field(
        default_factory=lambda: Path(
            os.getenv("CNG_EVAL_TASK_POOL", "data/sample_eval_tasks.json")
        ).resolve()
    )
    cors_origins: List[str] = field(
        default_factory=lambda: _parse_csv_env(
            "CNG_CORS_ORIGINS", ["http://localhost:5173", "http://127.0.0.1:5173"]
        )
    )
    preload_models: List[str] = field(
        default_factory=lambda: _parse_csv_env("CNG_PRELOAD_MODELS", list(SUPPORTED_MODELS))
    )
    required_models: List[str] = field(
        default_factory=lambda: _parse_csv_env("CNG_REQUIRED_MODELS", list(SUPPORTED_MODELS))
    )
    disabled_models: List[str] = field(
        default_factory=lambda: _parse_csv_env("CNG_DISABLED_MODELS", [])
    )
    device_preference: str = field(default_factory=lambda: os.getenv("CNG_DEVICE", "cpu").lower())
    default_temperature: float = field(
        default_factory=lambda: float(os.getenv("CNG_DEFAULT_TEMPERATURE", "0.8"))
    )
    enable_dynasty_mode: bool = field(
        default_factory=lambda: _parse_bool_env("CNG_ENABLE_DYNASTY_MODE", True)
    )
    feedback_database_url: Optional[str] = field(
        default_factory=lambda: os.getenv("CNG_FEEDBACK_DATABASE_URL")
    )
    feedback_auto_init: bool = field(
        default_factory=lambda: _parse_bool_env("CNG_FEEDBACK_AUTO_INIT", False)
    )
    generation_rate_limit_per_minute: int = field(
        default_factory=lambda: _parse_int_env("CNG_GENERATION_RATE_LIMIT_PER_MINUTE", 60)
    )
    feedback_rate_limit_per_minute: int = field(
        default_factory=lambda: _parse_int_env("CNG_FEEDBACK_RATE_LIMIT_PER_MINUTE", 30)
    )
    app_version: str = field(default_factory=lambda: os.getenv("CNG_APP_VERSION", "1.1.0"))
    public_run_id: str = field(default_factory=lambda: os.getenv("CNG_PUBLIC_RUN_ID", "portfolio_demo_v1"))

    def resolve_device(self) -> str:
        if self.device_preference == "auto":
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        if self.device_preference == "cuda":
            import torch

            if not torch.cuda.is_available():
                LOGGER.warning("CUDA requested but unavailable; falling back to CPU.")
                return "cpu"
        return self.device_preference

    def validate(self) -> None:
        for collection_name, collection in (
            ("preload_models", self.preload_models),
            ("required_models", self.required_models),
            ("disabled_models", self.disabled_models),
        ):
            invalid = [model_type for model_type in collection if model_type not in SUPPORTED_MODELS]
            if invalid:
                raise RuntimeError(f"Invalid model names in {collection_name}: {invalid}")


@dataclass
class LoadedModel:
    model_type: str
    instance: Any
    source: str
    vocab: Any = None


@dataclass
class RuntimeState:
    settings: AppSettings
    device: str
    converter: opencc.OpenCC
    loaded_models: Dict[str, LoadedModel]
    model_statuses: Dict[str, Dict[str, Any]]
    task_pool: EvaluationTaskPool
    feedback_store: BaseFeedbackStore
    generation_rate_limiter: InMemoryRateLimiter
    feedback_rate_limiter: InMemoryRateLimiter
    dynasty_cache: Dict[int, LoadedModel]
    reference_full_names: List[str]
    constraint_support_cache: Dict[Tuple[str, str], int]
    fim_status: Dict[str, Any]
    fim_model: Optional[LoadedModel] = None
    model_load_lock: threading.Lock = field(default_factory=threading.Lock)


class GenerationRequest(BaseModel):
    model_type: str = Field(..., description="One of: markov, lstm, transformer")
    count: int = Field(default=5, ge=1, le=20)
    temperature: Optional[float] = Field(default=None, gt=0.0, le=2.0)
    seed: Optional[str] = Field(default=None, max_length=2)
    constraint_text: Optional[str] = Field(default=None, max_length=2)
    constraint_position: Literal["any", "start", "middle", "end"] = Field(default="any")

    @field_validator("model_type")
    @classmethod
    def normalize_model_type(cls, value: str) -> str:
        normalized = value.lower()
        if normalized not in SUPPORTED_MODELS:
            raise ValueError(f"model_type must be one of {SUPPORTED_MODELS}")
        return normalized

    @field_validator("seed")
    @classmethod
    def validate_seed(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None

    @field_validator("constraint_text")
    @classmethod
    def validate_constraint_text(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None

    @model_validator(mode="after")
    def validate_markov_temperature(self) -> "GenerationRequest":
        if self.model_type == "markov" and self.temperature is not None:
            raise ValueError("temperature is only supported for neural models.")
        if self.constraint_text is None and self.constraint_position != "any":
            raise ValueError("constraint_position requires constraint_text.")
        return self


class HistoricalPatternRequest(BaseModel):
    model_type: str = Field(..., description="One of: markov, lstm, transformer")
    fixed_token: str = Field(..., min_length=1, max_length=2)
    position: Literal["any", "start", "middle", "end"] = Field(default="any")
    count: int = Field(default=5, ge=1, le=10)
    seed: Optional[str] = Field(default=None, max_length=2)
    temperature: Optional[float] = Field(default=None, gt=0.0, le=2.0)

    @field_validator("model_type")
    @classmethod
    def normalize_model_type(cls, value: str) -> str:
        normalized = value.lower()
        if normalized not in SUPPORTED_MODELS:
            raise ValueError(f"model_type must be one of {SUPPORTED_MODELS}")
        return normalized

    @field_validator("fixed_token", "seed")
    @classmethod
    def strip_text(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None

    @model_validator(mode="after")
    def validate_markov_temperature(self) -> "HistoricalPatternRequest":
        if self.model_type == "markov" and self.temperature is not None:
            raise ValueError("temperature is only supported for neural models.")
        return self


class InfillRequest(BaseModel):
    fixed_token: str = Field(..., min_length=1, max_length=2)
    position: Literal["any", "start", "middle", "end"] = Field(default="any")
    count: int = Field(default=5, ge=1, le=10)
    seed: Optional[str] = Field(default=None, max_length=2)
    temperature: float = Field(default=0.8, gt=0.0, le=2.0)

    @field_validator("fixed_token", "seed")
    @classmethod
    def strip_text(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None


class CompareRequest(BaseModel):
    count: int = Field(default=5, ge=1, le=20)
    temperature: Optional[float] = Field(default=None, gt=0.0, le=2.0)
    seed: Optional[str] = Field(default=None, max_length=2)

    @field_validator("seed")
    @classmethod
    def validate_seed(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None


class DynastyGenerationRequest(BaseModel):
    dynasty_id: int = Field(..., description="One of the curated dynasty ids.")
    count: int = Field(default=5, ge=1, le=20)
    seed: Optional[str] = Field(default=None, max_length=2)
    constraint_text: Optional[str] = Field(default=None, max_length=2)
    constraint_position: Literal["any", "start", "middle", "end"] = Field(default="any")

    @field_validator("dynasty_id")
    @classmethod
    def validate_dynasty_id(cls, value: int) -> int:
        if value not in CURATED_DYNASTY_IDS:
            raise ValueError(f"dynasty_id must be one of {CURATED_DYNASTY_IDS}")
        return value

    @field_validator("seed")
    @classmethod
    def validate_seed(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None

    @field_validator("constraint_text")
    @classmethod
    def validate_constraint_text(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None

    @model_validator(mode="after")
    def validate_constraint_shape(self) -> "DynastyGenerationRequest":
        if self.constraint_text is None and self.constraint_position != "any":
            raise ValueError("constraint_position requires constraint_text.")
        return self


class GenerationMetadata(BaseModel):
    requested_count: int
    normalized_seed: Optional[str] = None
    seed_was_normalized: bool = False
    normalized_constraint_text: Optional[str] = None
    constraint_text_was_normalized: bool = False
    constraint_position: Optional[str] = None
    constraint_match_count: Optional[int] = None
    constraint_support_count: Optional[int] = None
    sampling_attempts: Optional[int] = None
    search_exhausted: bool = False
    warning: Optional[str] = None
    temperature: Optional[float] = None
    source: str
    dynasty_id: Optional[int] = None
    dynasty_code: Optional[str] = None
    dynasty_label_en: Optional[str] = None
    dynasty_label_zh: Optional[str] = None


class GenerationResponse(BaseModel):
    model_used: str
    generations: List[str]
    metadata: GenerationMetadata


class HistoricalPatternResponse(BaseModel):
    model_used: str
    generations: List[str]
    normalized_fixed_token: str
    fixed_token_was_normalized: bool
    position: Literal["any", "start", "middle", "end"]
    support_count: int
    support_level: Literal["strong", "weak", "none"]
    sampling_attempts: int
    search_exhausted: bool
    evidence_message: str
    metadata: GenerationMetadata


class InfillResponse(BaseModel):
    generations: List[str]
    normalized_fixed_token: str
    fixed_token_was_normalized: bool
    position: Literal["any", "start", "middle", "end"]
    constraint_satisfaction_rate: float
    historical_support_level: Literal["strong", "weak", "none"]
    creative_mode_notice: str


class ModelStatusResponse(BaseModel):
    model_type: str
    available: bool
    source: Optional[str] = None
    detail: str


class DynastyOptionResponse(BaseModel):
    dynasty_id: int
    code: str
    label_en: str
    label_zh: str
    available: bool
    detail: str


class CompareCandidateResponse(BaseModel):
    model_type: str
    generations: List[str]
    metadata: GenerationMetadata


class CompareResponse(BaseModel):
    run_id: str
    results: List[CompareCandidateResponse]
    request_metadata: GenerationMetadata


class EvalTaskResponse(BaseModel):
    task_id: str
    task_type: Literal["blind_pairwise", "dynasty_guess"]
    prompt_type: Optional[str] = None
    title: Optional[str] = None
    prompt: Optional[str] = None
    presented_payload: Dict[str, Any]
    response_options: List[str]


class FeedbackRequest(BaseModel):
    session_id: str = Field(..., min_length=6, max_length=128)
    locale: str = Field(..., min_length=2, max_length=16)
    surface: Literal["playground", "evaluation_lab"]
    task_type: Literal["favorite_pick", "blind_pairwise", "dynasty_guess"]
    task_id: Optional[str] = Field(default=None, max_length=128)
    run_id: Optional[str] = Field(default=None, max_length=128)
    checkpoint_id: Optional[str] = Field(default=None, max_length=128)
    prompt_type: Optional[str] = Field(default=None, max_length=128)
    presented_payload: Dict[str, Any] = Field(default_factory=dict)
    response_payload: Dict[str, Any] = Field(default_factory=dict)
    response_label: str = Field(..., min_length=1, max_length=64)
    latency_ms: int = Field(default=0, ge=0, le=600000)

    @model_validator(mode="after")
    def validate_task_shape(self) -> "FeedbackRequest":
        if self.task_type in {"blind_pairwise", "dynasty_guess"} and not self.task_id:
            raise ValueError("task_id is required for evaluation-lab feedback.")

        if self.task_type == "favorite_pick" and self.response_label not in SUPPORTED_MODELS:
            raise ValueError(f"favorite_pick response_label must be one of {SUPPORTED_MODELS}.")

        if self.task_type == "blind_pairwise" and self.response_label not in BLIND_PAIRWISE_CHOICES:
            raise ValueError(f"blind_pairwise response_label must be one of {BLIND_PAIRWISE_CHOICES}.")

        if self.task_type == "dynasty_guess" and self.response_label not in DYNASTY_GUESS_CHOICES:
            raise ValueError(f"dynasty_guess response_label must be one of {DYNASTY_GUESS_CHOICES}.")

        return self


class FeedbackResponse(BaseModel):
    status: str
    event_id: str
    detail: str


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _status(available: bool, detail: str, source: Optional[str] = None) -> Dict[str, Any]:
    return {"available": available, "detail": detail, "source": source}


def _model_artifact_status(settings: AppSettings, model_type: str) -> Dict[str, Any]:
    if model_type in settings.disabled_models:
        return _status(
            False,
            "Model is disabled in this deployment to stay within the public demo memory budget.",
        )

    if model_type == "markov":
        if settings.markov_artifact_path.exists():
            return _status(True, "Markov artifact is present and will load on demand.", str(settings.markov_artifact_path))
        if settings.db_path.exists():
            return _status(True, "CBDB database is present; Markov model can be built on demand.", str(settings.db_path))
        if settings.sample_data_path.exists():
            return _status(
                True,
                "Sample data is present; fallback Markov model can be built on demand.",
                str(settings.sample_data_path),
            )
        return _status(False, "No Markov artifact, CBDB database, or sample dataset was found.")

    if model_type == "lstm":
        missing = [
            str(path)
            for path in (settings.lstm_weights_path, settings.lstm_vocab_path)
            if not path.exists()
        ]
        if missing:
            return _status(False, f"LSTM artifact is missing: {', '.join(missing)}.")
        return _status(True, "LSTM artifacts are present and will load on demand.", str(settings.lstm_weights_path))

    if model_type == "transformer":
        missing = [
            str(path)
            for path in (settings.transformer_weights_path, settings.transformer_vocab_path)
            if not path.exists()
        ]
        if missing:
            return _status(False, f"Transformer artifact is missing: {', '.join(missing)}.")
        return _status(
            True,
            "Transformer artifacts are present and will load on demand.",
            str(settings.transformer_weights_path),
        )

    return _status(False, f"Unsupported model type: {model_type}.")


def _load_sample_names(sample_data_path: Path) -> tuple[List[str], List[str]]:
    if not sample_data_path.exists():
        return [], []

    payload = json.loads(sample_data_path.read_text(encoding="utf-8"))
    surnames = payload.get("surnames", [])
    given_names = payload.get("given_names", [])
    if not isinstance(surnames, list) or not isinstance(given_names, list):
        return [], []
    return surnames, given_names


def _load_reference_name_pairs(settings: AppSettings) -> tuple[List[str], List[str]]:
    surnames, given_names = get_cbdb_names(db_path=str(settings.db_path))
    if surnames and given_names:
        return surnames, given_names
    return _load_sample_names(settings.sample_data_path)


def _load_lstm(settings: AppSettings, device: str) -> LoadedModel:
    import torch
    from model import ChineseNameLSTM

    vocab = _load_pickle(settings.lstm_vocab_path)
    model = ChineseNameLSTM(
        vocab_size=len(vocab),
        embedding_dim=256,
        hidden_size=512,
        num_layers=2,
        pad_idx=vocab.PAD_IDX,
        dropout_prob=0.0,
    ).to(device)
    model.load_state_dict(torch.load(settings.lstm_weights_path, map_location=device, weights_only=True))
    model.eval()
    return LoadedModel("lstm", model, source="weights+vocab", vocab=vocab)


def _load_transformer(settings: AppSettings, device: str) -> LoadedModel:
    import torch
    from model import ChineseNameTransformer

    vocab = _load_pickle(settings.transformer_vocab_path)
    model = ChineseNameTransformer(
        vocab_size=len(vocab),
        embedding_dim=256,
        num_heads=8,
        hidden_dim=512,
        num_layers=2,
        max_seq_len=10,
        pad_idx=vocab.PAD_IDX,
        dropout_prob=0.0,
    ).to(device)
    model.load_state_dict(
        torch.load(settings.transformer_weights_path, map_location=device, weights_only=True)
    )
    model.eval()
    return LoadedModel("transformer", model, source="weights+vocab", vocab=vocab)


def _load_fim(settings: AppSettings, device: str) -> LoadedModel:
    import torch
    from model import ChineseNameTransformer

    checkpoint = torch.load(settings.fim_weights_path, map_location=device, weights_only=False)
    if checkpoint.get("artifact_type") != "fim_template_decoder":
        raise RuntimeError("FIM checkpoint is not a trained fim_template_decoder artifact.")

    config = checkpoint["config"]
    vocab = checkpoint.get("vocab") or _load_pickle(settings.fim_vocab_path)
    model = ChineseNameTransformer(
        vocab_size=len(vocab),
        embedding_dim=int(config["embedding_dim"]),
        num_heads=int(config["num_heads"]),
        hidden_dim=int(config["hidden_dim"]),
        num_layers=int(config["num_layers"]),
        max_seq_len=int(config["max_seq_len"]),
        pad_idx=vocab.PAD_IDX,
        dropout_prob=0.0,
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return LoadedModel(
        "fim_template",
        model,
        source=f"fim-template:{settings.fim_weights_path.name}",
        vocab={
            "vocab": vocab,
            "allowed_output_indices": checkpoint.get("allowed_output_indices"),
            "config": config,
        },
    )


def _build_markov_from_pairs(surnames: List[str], given_names: List[str], source: str) -> LoadedModel:
    if not surnames or not given_names:
        raise RuntimeError("No name pairs available to build the Markov model.")

    model = ChineseNameMarkov(surnames_list=surnames, boost_compound=False)
    model.train(given_names)
    return LoadedModel("markov", model, source=source)


def _load_markov(settings: AppSettings) -> LoadedModel:
    if settings.markov_artifact_path.exists():
        model = _load_pickle(settings.markov_artifact_path)
        return LoadedModel("markov", model, source="artifact")

    surnames, given_names = get_cbdb_names(db_path=str(settings.db_path))
    if surnames and given_names:
        return _build_markov_from_pairs(surnames, given_names, source="cbdb-runtime-build")

    sample_surnames, sample_given_names = _load_sample_names(settings.sample_data_path)
    if sample_surnames and sample_given_names:
        return _build_markov_from_pairs(sample_surnames, sample_given_names, source="sample-runtime-build")

    raise RuntimeError("No Markov artifact, CBDB database, or sample dataset was available.")


def _load_model_by_type(settings: AppSettings, device: str, model_type: str) -> LoadedModel:
    if model_type == "markov":
        return _load_markov(settings)
    if model_type == "lstm":
        return _load_lstm(settings, device)
    if model_type == "transformer":
        return _load_transformer(settings, device)
    raise RuntimeError(f"Unsupported model type: {model_type}.")


def _ensure_model_loaded(runtime: RuntimeState, model_type: str) -> LoadedModel:
    if model_type in runtime.settings.disabled_models:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Requested model '{model_type}' is disabled in this deployment to stay within "
                "the public demo memory budget."
            ),
        )

    model_entry = runtime.loaded_models.get(model_type)
    if model_entry is not None:
        return model_entry

    with runtime.model_load_lock:
        model_entry = runtime.loaded_models.get(model_type)
        if model_entry is not None:
            return model_entry

        LOGGER.info("Lazy loading model '%s'...", model_type)
        try:
            loaded_model = _load_model_by_type(runtime.settings, runtime.device, model_type)
        except Exception as exc:  # noqa: BLE001
            detail = f"{type(exc).__name__}: {exc}"
            LOGGER.exception("Failed to lazy load model '%s': %s", model_type, detail)
            runtime.model_statuses[model_type] = _status(False, detail)
            raise HTTPException(
                status_code=503,
                detail=f"Requested model '{model_type}' is unavailable: {detail}",
            ) from exc

        runtime.loaded_models[model_type] = loaded_model
        runtime.model_statuses[model_type] = _status(
            True,
            "Model loaded successfully.",
            loaded_model.source,
        )
        return loaded_model


def _ensure_fim_loaded(runtime: RuntimeState) -> LoadedModel:
    if not runtime.fim_status.get("available", False):
        detail = runtime.fim_status.get("detail") or "FIM model artifact is not available in this deployment."
        if "FIM model artifact is not available" not in detail:
            detail = f"FIM model artifact is not available in this deployment. {detail}"
        raise HTTPException(status_code=503, detail=detail)
    if runtime.fim_model is not None:
        return runtime.fim_model

    with runtime.model_load_lock:
        if runtime.fim_model is not None:
            return runtime.fim_model
        LOGGER.info("Lazy loading FIM model...")
        try:
            runtime.fim_model = _load_fim(runtime.settings, runtime.device)
        except Exception as exc:  # noqa: BLE001
            detail = f"{type(exc).__name__}: {exc}"
            LOGGER.exception("Failed to lazy load FIM model: %s", detail)
            runtime.fim_status = {**runtime.fim_status, "available": False, "detail": detail}
            raise HTTPException(status_code=503, detail=detail) from exc

        runtime.fim_status = {
            **runtime.fim_status,
            "available": True,
            "detail": "FIM model loaded successfully.",
        }
        return runtime.fim_model


def _resolve_client_identifier(request: Request) -> str:
    forwarded_for = request.headers.get("x-forwarded-for", "").strip()
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    if request.client and request.client.host:
        return request.client.host
    return "unknown-client"


def _enforce_rate_limit(
    request: Request,
    limiter: InMemoryRateLimiter,
    limit: int,
    bucket_name: str,
) -> None:
    client_identifier = _resolve_client_identifier(request)
    key = f"{bucket_name}:{client_identifier}"
    if not limiter.allow(key, limit):
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Please try again later.")


def _curated_dynasty_entry(dynasty_id: int) -> Dict[str, Any]:
    for dynasty in CURATED_DYNASTIES:
        if dynasty["dynasty_id"] == dynasty_id:
            return dict(dynasty)
    raise KeyError(f"Unknown curated dynasty id '{dynasty_id}'.")


def _dynasty_availability(settings: AppSettings, dynasty_id: int) -> Tuple[bool, str]:
    if dynasty_id not in CURATED_DYNASTY_IDS:
        return False, "Dynasty is not part of the curated public set."
    if not settings.enable_dynasty_mode:
        return False, "Dynasty mode is disabled by configuration."
    if not settings.db_path.exists():
        return False, "CBDB database is not available on this deployment."
    return True, "Dynasty mode is available."


def _get_or_build_dynasty_markov(runtime: RuntimeState, dynasty_id: int) -> LoadedModel:
    if dynasty_id in runtime.dynasty_cache:
        return runtime.dynasty_cache[dynasty_id]

    available, detail = _dynasty_availability(runtime.settings, dynasty_id)
    if not available:
        raise RuntimeError(detail)

    surnames, given_names = get_cbdb_names(db_path=str(runtime.settings.db_path), dynasty_id=dynasty_id)
    if not surnames or not given_names:
        raise RuntimeError(f"No CBDB name pairs were available for dynasty {dynasty_id}.")

    dynasty_info = _curated_dynasty_entry(dynasty_id)
    loaded_model = _build_markov_from_pairs(
        surnames=surnames,
        given_names=given_names,
        source=f"dynasty-runtime-build:{dynasty_info['code']}",
    )
    runtime.dynasty_cache[dynasty_id] = loaded_model
    return loaded_model


def _generate_from_model(
    *,
    model_entry: LoadedModel,
    model_type: str,
    count: int,
    normalized_seed: Optional[str],
    runtime: RuntimeState,
    temperature: Optional[float],
) -> Tuple[List[str], Optional[float]]:
    if model_type == "markov":
        generations = [
            model_entry.instance.generate(max_length=2, seed_surname=normalized_seed)
            for _ in range(count)
        ]
        return generations, None

    from inference import generate_name

    active_temperature = temperature or runtime.settings.default_temperature
    generations = [
        generate_name(
            model=model_entry.instance,
            vocab=model_entry.vocab,
            seed_text=normalized_seed,
            max_length=4,
            temperature=active_temperature,
            device=runtime.device,
            converter=runtime.converter,
        )
        for _ in range(count)
    ]
    return generations, active_temperature


def _count_constraint_support(runtime: RuntimeState, token: str, position: str) -> int:
    cache_key = (token, position)
    if cache_key in runtime.constraint_support_cache:
        return runtime.constraint_support_cache[cache_key]

    count = sum(
        1 for full_name in runtime.reference_full_names if token_matches_position(full_name, token, position)
    )
    runtime.constraint_support_cache[cache_key] = count
    return count


def _support_level(support_count: int) -> Literal["strong", "weak", "none"]:
    if support_count >= 100:
        return "strong"
    if support_count > 0:
        return "weak"
    return "none"


def _historical_evidence_message(
    token: str,
    position: str,
    support_count: int,
    support_level: str,
) -> str:
    position_phrase = "any position" if position == "any" else f"the {position} position"
    if support_level == "strong":
        return (
            f"The cleaned historical corpus contains {support_count} names with '{token}' in {position_phrase}. "
            "This is enough support for a pattern-search experience, but it is still evidence from this dataset, "
            "not a universal claim about historical naming."
        )
    if support_level == "weak":
        return (
            f"The cleaned historical corpus contains {support_count} names with '{token}' in {position_phrase}. "
            "Treat this as weak support: the generator can try the pattern, but failures are possible for low-frequency tokens."
        )
    return (
        f"The cleaned historical corpus contains no names with '{token}' in {position_phrase}. "
        "This does not prove that historical people never used it; it means the current cleaned evidence does not show "
        "a stable naming pattern for this request."
    )


def _creative_mode_notice(support_level: str) -> str:
    if support_level == "none":
        return (
            "Creative Ancient-Style mode is intended for constrained creative naming. "
            "Because historical support is currently absent, these outputs should not be read as historical reconstruction."
        )
    if support_level == "weak":
        return (
            "Creative Ancient-Style mode can explore a low-support token more flexibly, but the result remains creative "
            "generation rather than evidence of historical authenticity."
        )
    return (
        "Creative Ancient-Style mode is a controllable generation aid. Even with stronger corpus support, its outputs "
        "are creative suggestions rather than direct historical evidence."
    )


def _fim_status(settings: AppSettings) -> Dict[str, Any]:
    required_paths = {
        "weights": settings.fim_weights_path,
        "vocab": settings.fim_vocab_path,
        "config": settings.fim_config_path,
    }
    missing = [name for name, path in required_paths.items() if not path.exists()]
    available = bool(settings.enable_fim_mode and not missing)
    if not settings.enable_fim_mode:
        detail = "FIM mode is disabled by configuration."
    elif missing:
        detail = f"FIM model artifact is not available in this deployment. Missing: {', '.join(missing)}."
    else:
        detail = "FIM model artifacts are configured."
    return {
        "enabled": settings.enable_fim_mode,
        "available": available,
        "detail": detail,
        "weights_path": str(settings.fim_weights_path),
        "vocab_path": str(settings.fim_vocab_path),
        "config_path": str(settings.fim_config_path),
    }


def _estimate_constraint_attempt_budget(
    *,
    runtime: RuntimeState,
    model_type: str,
    requested_count: int,
    support_count: Optional[int],
) -> int:
    base_budget = max(400, requested_count * 250)
    if not support_count or not runtime.reference_full_names:
        return base_budget

    support_rate = support_count / max(len(runtime.reference_full_names), 1)
    expected_attempts = int((requested_count / max(support_rate, 1e-9)) * 3.0)
    max_budget = 50000 if model_type == "markov" else 12000
    return min(max(base_budget, expected_attempts), max_budget)


def _generate_with_constraints(
    *,
    model_entry: LoadedModel,
    model_type: str,
    count: int,
    normalized_seed: Optional[str],
    normalized_constraint_text: Optional[str],
    constraint_position: str,
    runtime: RuntimeState,
    temperature: Optional[float],
) -> Tuple[List[str], Optional[float], int, Optional[int], bool, Optional[str]]:
    if not normalized_constraint_text:
        generations, active_temperature = _generate_from_model(
            model_entry=model_entry,
            model_type=model_type,
            count=count,
            normalized_seed=normalized_seed,
            runtime=runtime,
            temperature=temperature,
        )
        return generations, active_temperature, count, None, False, None

    support_count = _count_constraint_support(runtime, normalized_constraint_text, constraint_position)
    if support_count == 0:
        raise ValueError(
            "This token/position combination does not appear in the cleaned training corpus. "
            "Try a different position, a more common character, or use lucky mode."
        )

    matched_generations: List[str] = []
    sampling_attempts = 0
    active_temperature: Optional[float] = None
    max_attempts = _estimate_constraint_attempt_budget(
        runtime=runtime,
        model_type=model_type,
        requested_count=count,
        support_count=support_count,
    )

    while len(matched_generations) < count and sampling_attempts < max_attempts:
        generations, active_temperature = _generate_from_model(
            model_entry=model_entry,
            model_type=model_type,
            count=1,
            normalized_seed=normalized_seed,
            runtime=runtime,
            temperature=temperature,
        )
        sampling_attempts += 1
        candidate = generations[0]
        if token_matches_position(candidate, normalized_constraint_text, constraint_position):
            matched_generations.append(candidate)

    if not matched_generations:
        raise RuntimeError(
            "No generations satisfied the requested token constraint within the current search budget. "
            f"The cleaned training corpus contains {support_count} matching names. "
            "Try count=1, a looser position, or use lucky mode."
        )

    search_exhausted = len(matched_generations) < count
    warning = None
    if search_exhausted:
        warning = (
            f"Only found {len(matched_generations)} matches after {sampling_attempts} attempts. "
            f"The cleaned training corpus contains {support_count} matching names for this constraint."
        )

    return matched_generations, active_temperature, sampling_attempts, support_count, search_exhausted, warning


def _public_task_counts(task_pool: EvaluationTaskPool) -> Dict[str, int]:
    return task_pool.list_task_types()


def load_runtime(
    settings: Optional[AppSettings] = None,
    *,
    feedback_store: Optional[BaseFeedbackStore] = None,
    task_pool: Optional[EvaluationTaskPool] = None,
) -> RuntimeState:
    active_settings = settings or AppSettings()
    active_settings.validate()
    converter = opencc.OpenCC("s2t")
    device = active_settings.resolve_device()

    loaded_models: Dict[str, LoadedModel] = {}
    model_statuses: Dict[str, Dict[str, Any]] = {
        model_type: _model_artifact_status(active_settings, model_type)
        for model_type in SUPPORTED_MODELS
    }

    for model_type in active_settings.preload_models:
        if model_type in active_settings.disabled_models:
            LOGGER.info("Skipping disabled model '%s'.", model_type)
            continue
        LOGGER.info("Loading model '%s'...", model_type)
        try:
            loaded_model = _load_model_by_type(active_settings, device, model_type)
            loaded_models[model_type] = loaded_model
            model_statuses[model_type] = _status(True, "Model loaded successfully.", loaded_model.source)
        except Exception as exc:  # noqa: BLE001 - keep startup diagnostics concise
            detail = f"{type(exc).__name__}: {exc}"
            LOGGER.exception("Failed to load model '%s': %s", model_type, detail)
            model_statuses[model_type] = _status(False, detail)
            if model_type in active_settings.required_models:
                raise RuntimeError(f"Required model '{model_type}' failed to load: {detail}") from exc

    active_task_pool = task_pool or EvaluationTaskPool(active_settings.eval_task_pool_path)
    active_feedback_store = feedback_store or build_feedback_store(
        database_url=active_settings.feedback_database_url,
        auto_init=active_settings.feedback_auto_init,
    )
    reference_surnames, reference_given_names = _load_reference_name_pairs(active_settings)
    reference_full_names = [
        f"{surname}{given_name}"
        for surname, given_name in zip(reference_surnames, reference_given_names)
        if surname and given_name
    ]
    fim_status = _fim_status(active_settings)
    fim_model: Optional[LoadedModel] = None
    if fim_status["available"]:
        fim_status = {**fim_status, "detail": "FIM artifacts are present and will load on demand."}

    return RuntimeState(
        settings=active_settings,
        device=device,
        converter=converter,
        loaded_models=loaded_models,
        model_statuses=model_statuses,
        task_pool=active_task_pool,
        feedback_store=active_feedback_store,
        generation_rate_limiter=InMemoryRateLimiter(),
        feedback_rate_limiter=InMemoryRateLimiter(),
        dynasty_cache={},
        reference_full_names=reference_full_names,
        constraint_support_cache={},
        fim_status=fim_status,
        fim_model=fim_model,
    )


def create_app(
    settings: Optional[AppSettings] = None,
    *,
    feedback_store: Optional[BaseFeedbackStore] = None,
    task_pool: Optional[EvaluationTaskPool] = None,
) -> FastAPI:
    active_settings = settings or AppSettings()

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        LOGGER.info("Starting Chinese Name Generator API...")
        runtime = load_runtime(active_settings, feedback_store=feedback_store, task_pool=task_pool)
        app.state.runtime = runtime
        LOGGER.info("API startup complete. Loaded models: %s", sorted(runtime.loaded_models.keys()))
        LOGGER.info("Task pool counts: %s", _public_task_counts(runtime.task_pool))
        LOGGER.info("Feedback store status: %s", runtime.feedback_store.detail())
        yield
        LOGGER.info("Shutting down Chinese Name Generator API.")
        app.state.runtime = None

    app = FastAPI(
        title="Chinese Name Generator API",
        version="1.1.0",
        description="A comparative inference API for Markov, LSTM, Transformer, and public human-feedback tasks.",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=active_settings.cors_origins or ["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(RequestValidationError)
    async def request_validation_exception_handler(_, exc: RequestValidationError):
        return JSONResponse(
            status_code=400,
            content={
                "detail": "Invalid request payload.",
                "errors": jsonable_encoder(exc.errors()),
            },
        )

    @app.get("/health")
    async def health_check():
        runtime: RuntimeState = app.state.runtime
        missing_required = [
            model_type
            for model_type in runtime.settings.required_models
            if not runtime.model_statuses.get(model_type, {}).get("available", False)
        ]
        status = "ok" if not missing_required else "degraded"
        return {
            "status": status,
            "device": runtime.device,
            "loaded_models": sorted(runtime.loaded_models.keys()),
            "missing_required_models": missing_required,
            "dynasty_mode_enabled": runtime.settings.enable_dynasty_mode,
            "fim_mode": runtime.fim_status,
            "feedback_store_available": runtime.feedback_store.is_available(),
            "feedback_store_detail": runtime.feedback_store.detail(),
            "task_pool_counts": _public_task_counts(runtime.task_pool),
        }

    @app.get("/models", response_model=List[ModelStatusResponse])
    async def models():
        runtime: RuntimeState = app.state.runtime
        return [
            ModelStatusResponse(model_type=model_type, **runtime.model_statuses[model_type])
            for model_type in SUPPORTED_MODELS
        ]

    @app.get("/dynasties", response_model=List[DynastyOptionResponse])
    async def dynasties():
        runtime: RuntimeState = app.state.runtime
        payload = []
        for dynasty in CURATED_DYNASTIES:
            available, detail = _dynasty_availability(runtime.settings, dynasty["dynasty_id"])
            payload.append(
                DynastyOptionResponse(
                    dynasty_id=dynasty["dynasty_id"],
                    code=dynasty["code"],
                    label_en=dynasty["label_en"],
                    label_zh=dynasty["label_zh"],
                    available=available,
                    detail=detail,
                )
            )
        return payload

    @app.post("/generate", response_model=GenerationResponse)
    async def generate_names_endpoint(request: Request, payload: GenerationRequest):
        runtime: RuntimeState = app.state.runtime
        _enforce_rate_limit(
            request,
            runtime.generation_rate_limiter,
            runtime.settings.generation_rate_limit_per_minute,
            "generate",
        )
        model_entry = _ensure_model_loaded(runtime, payload.model_type)

        normalized_seed, seed_was_normalized = normalize_seed_text(payload.seed, converter=runtime.converter)
        normalized_constraint_text, constraint_text_was_normalized = normalize_prompt_text(
            payload.constraint_text, converter=runtime.converter
        )

        try:
            (
                generations,
                active_temperature,
                sampling_attempts,
                constraint_support_count,
                search_exhausted,
                warning,
            ) = _generate_with_constraints(
                model_entry=model_entry,
                model_type=payload.model_type,
                count=payload.count,
                normalized_seed=normalized_seed,
                normalized_constraint_text=normalized_constraint_text,
                constraint_position=payload.constraint_position,
                runtime=runtime,
                temperature=payload.temperature,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001 - do not leak stack traces to clients
            LOGGER.exception("Generation failed for model '%s': %s", payload.model_type, exc)
            if normalized_constraint_text:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            raise HTTPException(
                status_code=500,
                detail=f"Generation failed for model '{payload.model_type}'. Check server logs for details.",
            ) from exc

        metadata = GenerationMetadata(
            requested_count=payload.count,
            normalized_seed=normalized_seed,
            seed_was_normalized=seed_was_normalized,
            normalized_constraint_text=normalized_constraint_text,
            constraint_text_was_normalized=constraint_text_was_normalized,
            constraint_position=payload.constraint_position if normalized_constraint_text else None,
            constraint_match_count=len(generations) if normalized_constraint_text else None,
            constraint_support_count=constraint_support_count if normalized_constraint_text else None,
            sampling_attempts=sampling_attempts if normalized_constraint_text else payload.count,
            search_exhausted=search_exhausted if normalized_constraint_text else False,
            warning=warning,
            temperature=active_temperature,
            source=model_entry.source,
        )
        return GenerationResponse(
            model_used=payload.model_type,
            generations=generations,
            metadata=metadata,
        )

    @app.post("/generate/historical-pattern", response_model=HistoricalPatternResponse)
    async def generate_historical_pattern(request: Request, payload: HistoricalPatternRequest):
        runtime: RuntimeState = app.state.runtime
        _enforce_rate_limit(
            request,
            runtime.generation_rate_limiter,
            runtime.settings.generation_rate_limit_per_minute,
            "generate_historical_pattern",
        )
        model_entry = _ensure_model_loaded(runtime, payload.model_type)

        normalized_seed, seed_was_normalized = normalize_seed_text(payload.seed, converter=runtime.converter)
        normalized_fixed_token, fixed_token_was_normalized = normalize_prompt_text(
            payload.fixed_token, converter=runtime.converter
        )
        if not normalized_fixed_token:
            raise HTTPException(status_code=400, detail="fixed_token is required.")

        support_count = _count_constraint_support(runtime, normalized_fixed_token, payload.position)
        support_level = _support_level(support_count)
        evidence_message = _historical_evidence_message(
            normalized_fixed_token,
            payload.position,
            support_count,
            support_level,
        )

        if support_level == "none":
            metadata = GenerationMetadata(
                requested_count=payload.count,
                normalized_seed=normalized_seed,
                seed_was_normalized=seed_was_normalized,
                normalized_constraint_text=normalized_fixed_token,
                constraint_text_was_normalized=fixed_token_was_normalized,
                constraint_position=payload.position,
                constraint_match_count=0,
                constraint_support_count=support_count,
                sampling_attempts=0,
                search_exhausted=True,
                warning=evidence_message,
                temperature=None,
                source=model_entry.source,
            )
            return HistoricalPatternResponse(
                model_used=payload.model_type,
                generations=[],
                normalized_fixed_token=normalized_fixed_token,
                fixed_token_was_normalized=fixed_token_was_normalized,
                position=payload.position,
                support_count=support_count,
                support_level=support_level,
                sampling_attempts=0,
                search_exhausted=True,
                evidence_message=evidence_message,
                metadata=metadata,
            )

        try:
            (
                generations,
                active_temperature,
                sampling_attempts,
                constraint_support_count,
                search_exhausted,
                warning,
            ) = _generate_with_constraints(
                model_entry=model_entry,
                model_type=payload.model_type,
                count=payload.count,
                normalized_seed=normalized_seed,
                normalized_constraint_text=normalized_fixed_token,
                constraint_position=payload.position,
                runtime=runtime,
                temperature=payload.temperature,
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Historical pattern generation failed for model '%s': %s", payload.model_type, exc)
            raise HTTPException(
                status_code=422,
                detail=(
                    "The historical pattern search could not satisfy this token constraint within the current "
                    "sampling budget. Try a looser position, count=1, or Creative Ancient-Style mode."
                ),
            ) from exc

        metadata = GenerationMetadata(
            requested_count=payload.count,
            normalized_seed=normalized_seed,
            seed_was_normalized=seed_was_normalized,
            normalized_constraint_text=normalized_fixed_token,
            constraint_text_was_normalized=fixed_token_was_normalized,
            constraint_position=payload.position,
            constraint_match_count=len(generations),
            constraint_support_count=constraint_support_count,
            sampling_attempts=sampling_attempts,
            search_exhausted=search_exhausted,
            warning=warning,
            temperature=active_temperature,
            source=model_entry.source,
        )
        return HistoricalPatternResponse(
            model_used=payload.model_type,
            generations=generations,
            normalized_fixed_token=normalized_fixed_token,
            fixed_token_was_normalized=fixed_token_was_normalized,
            position=payload.position,
            support_count=support_count,
            support_level=support_level,
            sampling_attempts=sampling_attempts,
            search_exhausted=search_exhausted,
            evidence_message=evidence_message,
            metadata=metadata,
        )

    @app.post("/generate/infill", response_model=InfillResponse)
    async def generate_infill(request: Request, payload: InfillRequest):
        runtime: RuntimeState = app.state.runtime
        _enforce_rate_limit(
            request,
            runtime.generation_rate_limiter,
            runtime.settings.generation_rate_limit_per_minute,
            "generate_infill",
        )
        normalized_fixed_token, fixed_token_was_normalized = normalize_prompt_text(
            payload.fixed_token, converter=runtime.converter
        )
        if not normalized_fixed_token:
            raise HTTPException(status_code=400, detail="fixed_token is required.")

        support_count = _count_constraint_support(runtime, normalized_fixed_token, payload.position)
        support_level = _support_level(support_count)
        notice = _creative_mode_notice(support_level)

        fim_model = _ensure_fim_loaded(runtime)
        from inference import generate_fim_name

        normalized_seed, _ = normalize_seed_text(payload.seed, converter=runtime.converter)
        vocab_payload = fim_model.vocab
        vocab = vocab_payload["vocab"]
        allowed_output_indices = vocab_payload.get("allowed_output_indices")
        config = vocab_payload.get("config", {})
        max_new_tokens = min(int(config.get("max_new_tokens", 4)), 4)
        generations: List[str] = []
        for _ in range(payload.count):
            candidate = ""
            for _attempt in range(20):
                candidate = generate_fim_name(
                    model=fim_model.instance,
                    vocab=vocab,
                    fixed_token=normalized_fixed_token,
                    position=payload.position,
                    seed_text=normalized_seed,
                    max_new_tokens=max_new_tokens,
                    min_new_tokens=2,
                    temperature=payload.temperature,
                    device=runtime.device,
                    converter=runtime.converter,
                    allowed_output_indices=allowed_output_indices,
                )
                if token_matches_position(candidate, normalized_fixed_token, payload.position) and token_is_single_occurrence(
                    candidate, normalized_fixed_token
                ):
                    break
            generations.append(candidate)
        satisfied = sum(
            1
            for name in generations
            if token_matches_position(name, normalized_fixed_token, payload.position)
            and token_is_single_occurrence(name, normalized_fixed_token)
        )
        return InfillResponse(
            generations=generations,
            normalized_fixed_token=normalized_fixed_token,
            fixed_token_was_normalized=fixed_token_was_normalized,
            position=payload.position,
            constraint_satisfaction_rate=round(satisfied / max(len(generations), 1), 8),
            historical_support_level=support_level,
            creative_mode_notice=notice,
        )

    @app.post("/generate/dynasty", response_model=GenerationResponse)
    async def generate_dynasty_names(request: Request, payload: DynastyGenerationRequest):
        runtime: RuntimeState = app.state.runtime
        _enforce_rate_limit(
            request,
            runtime.generation_rate_limiter,
            runtime.settings.generation_rate_limit_per_minute,
            "generate_dynasty",
        )

        try:
            dynasty_model = _get_or_build_dynasty_markov(runtime, payload.dynasty_id)
        except Exception as exc:  # noqa: BLE001 - expose as controlled unavailability
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        dynasty_info = _curated_dynasty_entry(payload.dynasty_id)
        normalized_seed, seed_was_normalized = normalize_seed_text(payload.seed, converter=runtime.converter)
        normalized_constraint_text, constraint_text_was_normalized = normalize_prompt_text(
            payload.constraint_text, converter=runtime.converter
        )
        try:
            (
                generations,
                _,
                sampling_attempts,
                constraint_support_count,
                search_exhausted,
                warning,
            ) = _generate_with_constraints(
                model_entry=dynasty_model,
                model_type="markov",
                count=payload.count,
                normalized_seed=normalized_seed,
                normalized_constraint_text=normalized_constraint_text,
                constraint_position=payload.constraint_position,
                runtime=runtime,
                temperature=None,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Dynasty generation failed for dynasty '%s': %s", payload.dynasty_id, exc)
            if normalized_constraint_text:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            raise HTTPException(
                status_code=500,
                detail=f"Dynasty generation failed for '{dynasty_info['code']}'. Check server logs for details.",
            ) from exc

        metadata = GenerationMetadata(
            requested_count=payload.count,
            normalized_seed=normalized_seed,
            seed_was_normalized=seed_was_normalized,
            normalized_constraint_text=normalized_constraint_text,
            constraint_text_was_normalized=constraint_text_was_normalized,
            constraint_position=payload.constraint_position if normalized_constraint_text else None,
            constraint_match_count=len(generations) if normalized_constraint_text else None,
            constraint_support_count=constraint_support_count if normalized_constraint_text else None,
            sampling_attempts=sampling_attempts if normalized_constraint_text else payload.count,
            search_exhausted=search_exhausted if normalized_constraint_text else False,
            warning=warning,
            source=dynasty_model.source,
            dynasty_id=payload.dynasty_id,
            dynasty_code=dynasty_info["code"],
            dynasty_label_en=dynasty_info["label_en"],
            dynasty_label_zh=dynasty_info["label_zh"],
        )
        return GenerationResponse(model_used="markov", generations=generations, metadata=metadata)

    @app.post("/compare", response_model=CompareResponse)
    async def compare_models(request: Request, payload: CompareRequest):
        runtime: RuntimeState = app.state.runtime
        _enforce_rate_limit(
            request,
            runtime.generation_rate_limiter,
            runtime.settings.generation_rate_limit_per_minute,
            "compare",
        )

        normalized_seed, seed_was_normalized = normalize_seed_text(
            payload.seed, converter=runtime.converter
        )
        results: List[CompareCandidateResponse] = []
        for model_type in SUPPORTED_MODELS:
            if not runtime.model_statuses.get(model_type, {}).get("available", False):
                continue
            try:
                model_entry = _ensure_model_loaded(runtime, model_type)
                generations, active_temperature = _generate_from_model(
                    model_entry=model_entry,
                    model_type=model_type,
                    count=payload.count,
                    normalized_seed=normalized_seed,
                    runtime=runtime,
                    temperature=payload.temperature,
                )
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception("Compare generation failed for model '%s': %s", model_type, exc)
                continue

            results.append(
                CompareCandidateResponse(
                    model_type=model_type,
                    generations=generations,
                    metadata=GenerationMetadata(
                        requested_count=payload.count,
                        normalized_seed=normalized_seed,
                        seed_was_normalized=seed_was_normalized,
                        temperature=active_temperature,
                        source=model_entry.source,
                    ),
                )
            )

        if not results:
            raise HTTPException(status_code=503, detail="No models are available for comparison.")

        request_metadata = GenerationMetadata(
            requested_count=payload.count,
            normalized_seed=normalized_seed,
            seed_was_normalized=seed_was_normalized,
            temperature=payload.temperature or runtime.settings.default_temperature,
            source="compare-request",
        )
        return CompareResponse(
            run_id=runtime.settings.public_run_id,
            results=results,
            request_metadata=request_metadata,
        )

    @app.get("/eval/tasks", response_model=EvalTaskResponse)
    async def get_eval_task(
        task_type: Literal["blind_pairwise", "dynasty_guess"] = Query(..., description="Evaluation task type.")
    ):
        runtime: RuntimeState = app.state.runtime
        try:
            task = runtime.task_pool.get_task(task_type)
        except KeyError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return EvalTaskResponse(**task)

    @app.post("/feedback", response_model=FeedbackResponse)
    async def submit_feedback(request: Request, payload: FeedbackRequest):
        runtime: RuntimeState = app.state.runtime
        _enforce_rate_limit(
            request,
            runtime.feedback_rate_limiter,
            runtime.settings.feedback_rate_limit_per_minute,
            "feedback",
        )

        if not runtime.feedback_store.is_available():
            raise HTTPException(status_code=503, detail=runtime.feedback_store.detail())

        run_id = payload.run_id
        checkpoint_id = payload.checkpoint_id
        prompt_type = payload.prompt_type
        presented_payload = dict(payload.presented_payload)

        if payload.task_type in {"blind_pairwise", "dynasty_guess"}:
            try:
                task_metadata = runtime.task_pool.get_task_metadata(payload.task_id or "")
            except KeyError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

            run_id = task_metadata.get("run_id")
            checkpoint_id = task_metadata.get("checkpoint_id")
            prompt_type = task_metadata.get("prompt_type")
            presented_payload = dict(task_metadata.get("presented_payload", {}))

        if payload.task_type == "favorite_pick":
            run_id = run_id or runtime.settings.public_run_id
            checkpoint_id = checkpoint_id or "compare_live"
            prompt_type = prompt_type or "live_compare"

        event = StoredFeedbackEvent(
            session_id=payload.session_id,
            locale=payload.locale,
            surface=payload.surface,
            task_type=payload.task_type,
            task_id=payload.task_id,
            run_id=run_id,
            checkpoint_id=checkpoint_id,
            prompt_type=prompt_type,
            presented_payload=presented_payload,
            response_payload=dict(payload.response_payload),
            response_label=payload.response_label,
            latency_ms=payload.latency_ms,
            app_version=runtime.settings.app_version,
        )

        try:
            stored = runtime.feedback_store.record_event(event)
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Feedback submission failed: %s", exc)
            raise HTTPException(status_code=500, detail="Feedback submission failed. Check server logs for details.") from exc

        return FeedbackResponse(status="accepted", event_id=stored["id"], detail="Feedback recorded.")

    return app


app = create_app()
