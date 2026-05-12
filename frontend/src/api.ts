import type {
  CompareResponse,
  DynastyOption,
  EvalTask,
  FeedbackResponse,
  GenerationResponse,
  HistoricalPatternResponse,
  InfillResponse,
  ModelStatus,
  ModelType,
  SurfaceMode,
  Locale,
  ConstraintPosition,
} from "./types";

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL?.replace(/\/$/, "") ?? "http://127.0.0.1:8000";

interface GeneratePayload {
  model_type: ModelType;
  count: number;
  seed?: string;
  temperature?: number;
  constraint_text?: string;
  constraint_position?: ConstraintPosition;
}

interface DynastyGeneratePayload {
  dynasty_id: number;
  count: number;
  seed?: string;
  constraint_text?: string;
  constraint_position?: ConstraintPosition;
}

interface HistoricalPatternPayload {
  model_type: ModelType;
  fixed_token: string;
  position: ConstraintPosition;
  count: number;
  seed?: string;
  temperature?: number;
}

interface InfillPayload {
  fixed_token: string;
  position: ConstraintPosition;
  count: number;
  seed?: string;
  temperature: number;
}

interface ComparePayload {
  count: number;
  seed?: string;
  temperature?: number;
}

interface FeedbackPayload {
  session_id: string;
  locale: Locale;
  surface: SurfaceMode;
  task_type: "favorite_pick" | "blind_pairwise" | "dynasty_guess";
  task_id?: string;
  run_id?: string;
  checkpoint_id?: string;
  prompt_type?: string;
  presented_payload?: Record<string, unknown>;
  response_payload?: Record<string, unknown>;
  response_label: string;
  latency_ms: number;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {}),
    },
    ...init,
  });

  if (!response.ok) {
    let detail = `Request failed with status ${response.status}`;
    try {
      const payload = await response.json();
      if (typeof payload?.detail === "string") {
        detail = payload.detail;
      }
    } catch {
      // Keep the fallback error message.
    }
    throw new Error(detail);
  }

  return response.json() as Promise<T>;
}

export function fetchModels(): Promise<ModelStatus[]> {
  return request<ModelStatus[]>("/models");
}

export function fetchDynasties(): Promise<DynastyOption[]> {
  return request<DynastyOption[]>("/dynasties");
}

export function generateNames(payload: GeneratePayload): Promise<GenerationResponse> {
  return request<GenerationResponse>("/generate", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function generateDynastyNames(payload: DynastyGeneratePayload): Promise<GenerationResponse> {
  return request<GenerationResponse>("/generate/dynasty", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function generateHistoricalPattern(
  payload: HistoricalPatternPayload,
): Promise<HistoricalPatternResponse> {
  return request<HistoricalPatternResponse>("/generate/historical-pattern", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function generateInfill(payload: InfillPayload): Promise<InfillResponse> {
  return request<InfillResponse>("/generate/infill", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function compareModels(payload: ComparePayload): Promise<CompareResponse> {
  return request<CompareResponse>("/compare", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function fetchEvalTask(taskType: "blind_pairwise" | "dynasty_guess"): Promise<EvalTask> {
  return request<EvalTask>(`/eval/tasks?task_type=${encodeURIComponent(taskType)}`);
}

export function submitFeedback(payload: FeedbackPayload): Promise<FeedbackResponse> {
  return request<FeedbackResponse>("/feedback", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export { API_BASE_URL };
