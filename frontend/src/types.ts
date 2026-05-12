export type Locale = "en" | "zh";
export type SurfaceMode = "playground" | "evaluation_lab";
export type ModelType = "markov" | "lstm" | "transformer";
export type EvalTaskType = "blind_pairwise" | "dynasty_guess";
export type FavoriteTaskType = "favorite_pick";
export type FeedbackTaskType = FavoriteTaskType | EvalTaskType;
export type ConstraintPosition = "any" | "start" | "middle" | "end";
export type SupportLevel = "strong" | "weak" | "none";

export interface ModelStatus {
  model_type: ModelType;
  available: boolean;
  source?: string | null;
  detail: string;
}

export interface DynastyOption {
  dynasty_id: number;
  code: "Tang" | "Song" | "Ming" | "Qing";
  label_en: string;
  label_zh: string;
  available: boolean;
  detail: string;
}

export interface GenerationMetadata {
  requested_count: number;
  normalized_seed?: string | null;
  seed_was_normalized: boolean;
  normalized_constraint_text?: string | null;
  constraint_text_was_normalized?: boolean;
  constraint_position?: ConstraintPosition | null;
  constraint_match_count?: number | null;
  constraint_support_count?: number | null;
  sampling_attempts?: number | null;
  search_exhausted?: boolean;
  warning?: string | null;
  temperature?: number | null;
  source: string;
  dynasty_id?: number | null;
  dynasty_code?: string | null;
  dynasty_label_en?: string | null;
  dynasty_label_zh?: string | null;
}

export interface GenerationResponse {
  model_used: ModelType;
  generations: string[];
  metadata: GenerationMetadata;
}

export interface HistoricalPatternResponse {
  model_used: ModelType;
  generations: string[];
  normalized_fixed_token: string;
  fixed_token_was_normalized: boolean;
  position: ConstraintPosition;
  support_count: number;
  support_level: SupportLevel;
  sampling_attempts: number;
  search_exhausted: boolean;
  evidence_message: string;
  metadata: GenerationMetadata;
}

export interface InfillResponse {
  generations: string[];
  normalized_fixed_token: string;
  fixed_token_was_normalized: boolean;
  position: ConstraintPosition;
  constraint_satisfaction_rate: number;
  historical_support_level: SupportLevel;
  creative_mode_notice: string;
}

export interface CompareCandidate {
  model_type: ModelType;
  generations: string[];
  metadata: GenerationMetadata;
}

export interface CompareResponse {
  run_id: string;
  results: CompareCandidate[];
  request_metadata: GenerationMetadata;
}

export interface BlindPairwisePayload {
  left: {
    label: string;
    names: string[];
  };
  right: {
    label: string;
    names: string[];
  };
}

export interface DynastyGuessPayload {
  names: string[];
}

export interface EvalTask {
  task_id: string;
  task_type: EvalTaskType;
  prompt_type?: string | null;
  title?: string | null;
  prompt?: string | null;
  presented_payload: BlindPairwisePayload | DynastyGuessPayload;
  response_options: string[];
}

export interface FeedbackResponse {
  status: string;
  event_id: string;
  detail: string;
}
