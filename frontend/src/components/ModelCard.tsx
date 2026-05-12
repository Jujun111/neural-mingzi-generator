import type { Locale, ModelStatus } from "../types";

interface ModelCardProps {
  locale: Locale;
  model: ModelStatus;
  selected: boolean;
  onSelect: (modelType: ModelStatus["model_type"]) => void;
}

const descriptions = {
  en: {
    markov: "Fast probabilistic baseline with strong lexical diversity and very lightweight inference.",
    lstm: "Sequential recurrent model and the strongest performer in the current matched-pair study.",
    transformer: "Small causal decoder included to keep the architectural comparison concrete and honest.",
    live: "Live",
    unavailable: "Unavailable",
  },
  zh: {
    markov: "快速的概率基线模型，词汇风格多样，推理开销也最轻。",
    lstm: "顺序递归模型，也是当前 matched-pair 实验里表现最强的神经模型。",
    transformer: "小型 causal decoder，用来把架构差异放到一个具体而诚实的对比里。",
    live: "可用",
    unavailable: "不可用",
  },
} as const;

export function ModelCard({ locale, model, selected, onSelect }: ModelCardProps) {
  const text = descriptions[locale];

  return (
    <button
      type="button"
      className={`model-card ${selected ? "is-selected" : ""}`}
      onClick={() => onSelect(model.model_type)}
      disabled={!model.available}
    >
      <div className="model-card__header">
        <span className="model-card__title">{model.model_type.toUpperCase()}</span>
        <span className={`model-card__badge ${model.available ? "is-live" : "is-offline"}`}>
          {model.available ? text.live : text.unavailable}
        </span>
      </div>
      <p>{text[model.model_type]}</p>
      <p className="model-card__detail">{model.detail}</p>
    </button>
  );
}
