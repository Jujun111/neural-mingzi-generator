import { FormEvent, useEffect, useState } from "react";

import {
  compareModels,
  fetchDynasties,
  fetchEvalTask,
  fetchModels,
  generateDynastyNames,
  generateHistoricalPattern,
  generateInfill,
  generateNames,
  submitFeedback,
} from "./api";
import { ModelCard } from "./components/ModelCard";
import { ResultList } from "./components/ResultList";
import type {
  BlindPairwisePayload,
  CompareResponse,
  ConstraintPosition,
  DynastyGuessPayload,
  DynastyOption,
  EvalTask,
  EvalTaskType,
  GenerationResponse,
  HistoricalPatternResponse,
  InfillResponse,
  Locale,
  ModelStatus,
  ModelType,
  SurfaceMode,
} from "./types";

type WorkshopMode = "historical" | "creative";

const LOCALE_STORAGE_KEY = "cng.locale";
const SESSION_STORAGE_KEY = "cng.session_id";

const modelOrder: ModelType[] = ["markov", "lstm", "transformer"];
const constraintPositionOrder: ConstraintPosition[] = ["any", "start", "middle", "end"];
const REPO_URL = import.meta.env.VITE_GITHUB_URL ?? "https://github.com/";
const BLOG_URL = import.meta.env.VITE_BLOG_URL ?? "/technical_blog.md";

function createSessionId(): string {
  if (globalThis.crypto?.randomUUID) {
    return globalThis.crypto.randomUUID();
  }
  return `session-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function getInitialLocale(): Locale {
  const stored = localStorage.getItem(LOCALE_STORAGE_KEY);
  return stored === "zh" ? "zh" : "en";
}

function getOrCreateSessionId(): string {
  const stored = localStorage.getItem(SESSION_STORAGE_KEY);
  if (stored) {
    return stored;
  }
  const sessionId = createSessionId();
  localStorage.setItem(SESSION_STORAGE_KEY, sessionId);
  return sessionId;
}

function isBlindPairwisePayload(payload: unknown): payload is BlindPairwisePayload {
  if (payload === null || typeof payload !== "object") {
    return false;
  }
  return "left" in payload && "right" in payload;
}

function isDynastyGuessPayload(payload: unknown): payload is DynastyGuessPayload {
  if (payload === null || typeof payload !== "object") {
    return false;
  }
  return "names" in payload;
}

const copy = {
  en: {
    languageLabel: "Language",
    eyebrow: "Micro-sequence Generation Lab",
    title: "Chinese Name Generator",
    heroSummary:
      "A public-facing comparative ML demo that combines live generation, dynasty-style exploration, and lightweight human feedback collection.",
    surfaceLabel: "Mode",
    playground: "Playground",
    evaluationLab: "Evaluation Lab",
    playgroundSummary:
      "Explore the three generators openly, try dynasty-style Markov generation, and compare outputs side by side.",
    evaluationSummary:
      "Anonymous micro-tasks designed for lightweight but analyzable public feedback.",
    tabs: {
      classic: "Classic",
      dynasty: "Dynasty",
      compare: "Compare",
      workshop: "Name Workshop",
      blindPairwise: "Blind Pairwise",
      dynastyGuess: "Dynasty Guess",
    },
    classicTitle: "Classic Generator",
    classicSummary:
      "Select a model, optionally provide a surname seed, and generate a batch directly from the live API.",
    generatorTitle: "Generation Controls",
    selectedModel: "Selected Model",
    seed: "Seed",
    seedPlaceholder: "李 / 欧阳",
    count: "Count",
    temperature: "Temperature",
    markovTemperature: "Not used by Markov",
    generate: "Generate",
    generating: "Generating...",
    resultsTitle: "Results",
    resultsSummary: "Live inference output with normalized prompt metadata.",
    emptyState: "Pick a setup and generate a batch to see names here.",
    source: "Source",
    seedMeta: "Seed",
    none: "None",
    normalizedSuffix: " (normalized to Traditional Chinese)",
    countMeta: "Count",
    dynastyTitle: "Dynasty Generator",
    dynastySummary:
      "Markov-only mode trained from CBDB dynasty slices. Treat it as a style exploration surface rather than a formal historical authenticity claim.",
    dynastyLabel: "Dynasty",
    dynastyUnavailable: "Dynasty mode is unavailable on this deployment.",
    dynastyHint:
      "This mode currently uses dynasty-specific Markov models built from private CBDB data on the backend.",
    compareTitle: "Model Compare",
    compareSummary:
      "Generate side-by-side outputs from all available models using the same prompt settings, then pick the one you prefer.",
    favoritePick: "Pick Favorite",
    favoriteSaved: "Preference saved.",
    compareEmpty: "Run a compare request to see all models side by side.",
    workshopTitle: "Name Workshop",
    workshopSummary:
      "For users who already have a favorite character or phrase, split the intent into historical pattern search and creative ancient-style completion.",
    workshopHistorical: "Historical Pattern",
    workshopCreative: "Creative Ancient-Style",
    workshopIntroHistorical:
      "Evidence-based constrained search over the cleaned historical corpus. It answers whether the current data shows a usable naming pattern.",
    workshopIntroCreative:
      "A template Fill-in-the-Middle route for creative completion. It is designed for controllability, not historical reconstruction.",
    fixedTokenRequired: "Enter a fixed token first.",
    supportLevel: "Evidence level",
    evidenceMessage: "Evidence note",
    satisfactionRate: "Constraint satisfaction",
    creativeUnavailable:
      "FIM model artifact is not available in this deployment. Historical Pattern mode is still usable.",
    creativeNotice:
      "Creative mode is not a historical authenticity claim; it is constrained ancient-style generation.",
    tryCreative: "Try Creative Ancient-Style Mode",
    blindPairwiseTitle: "Blind Pairwise",
    blindPairwiseSummary:
      "Two anonymous batches, one choice. This is the main lightweight public human-feedback task.",
    blindPrompt: "Which batch feels more like plausible Chinese names?",
    dynastyGuessTitle: "Dynasty Guess",
    dynastyGuessSummary:
      "A small style-recognition task based on dynasty-conditioned Markov outputs.",
    dynastyGuessPrompt: "Which dynasty does this batch feel closest to?",
    batchA: "Batch A",
    batchB: "Batch B",
    nextTask: "Load Another Task",
    refreshTask: "Refresh Task",
    answers: {
      left: "Choose A",
      right: "Choose B",
      tie: "Tie",
      skip: "Skip",
      Tang: "Tang",
      Song: "Song",
      Ming: "Ming",
      Qing: "Qing",
    },
    feedbackAccepted: "Feedback recorded. Loading the next task...",
    feedbackUnavailable: "Feedback storage is not available right now.",
    loadingTask: "Loading task...",
    notesTitle: "Architecture Notes",
    notesSummary:
      "The public app keeps weaker and stronger models visible so the engineering and research story stays honest.",
    notes: {
      markov: "Great for explainability, fast inference, and dynasty-style exploration.",
      lstm: "Still the strongest performer in the current fairness-first matched-pair study.",
      transformer: "Included as a real deployed comparator rather than an abstract claim.",
    },
    github: "GitHub Repo",
    blog: "Technical Blog",
    connectError: "Failed to connect to the API.",
    generationError: "Generation failed.",
  },
  zh: {
    languageLabel: "语言",
    eyebrow: "中文姓名微序列生成实验室",
    title: "中文姓名生成器",
    heroSummary:
      "一个面向公开展示的比较式 ML demo，把实时生成、朝代风格探索和轻量人类反馈采集放到同一个 Web app 里。",
    surfaceLabel: "模式",
    playground: "试玩区",
    evaluationLab: "评测区",
    playgroundSummary:
      "公开体验三种生成器，试玩朝代风格 Markov 模式，并把不同模型的结果并排比较。",
    evaluationSummary:
      "匿名微任务界面，用来收集轻量但可分析的真实用户反馈。",
    tabs: {
      classic: "经典生成",
      dynasty: "朝代玩法",
      compare: "模型对比",
      blindPairwise: "匿名双选",
      dynastyGuess: "朝代猜测",
    },
    classicTitle: "经典生成器",
    classicSummary: "选择模型，可选输入姓氏种子，然后直接从在线 API 生成一批名字。",
    generatorTitle: "生成控制",
    selectedModel: "当前模型",
    seed: "种子",
    seedPlaceholder: "李 / 欧阳",
    count: "数量",
    temperature: "温度",
    markovTemperature: "Markov 不使用温度",
    generate: "生成",
    generating: "生成中...",
    resultsTitle: "结果",
    resultsSummary: "实时推理输出，附带种子归一化等元信息。",
    emptyState: "先选择一组参数并生成名字，这里会显示结果。",
    source: "来源",
    seedMeta: "种子",
    none: "无",
    normalizedSuffix: "（已归一化为繁体中文）",
    countMeta: "数量",
    dynastyTitle: "朝代生成器",
    dynastySummary:
      "Markov 专属模式，按 CBDB 的朝代切片构建。它更适合作为风格探索玩法，而不是历史真实性的强结论。",
    dynastyLabel: "朝代",
    dynastyUnavailable: "当前部署没有启用朝代模式。",
    dynastyHint: "这个模式目前依赖后端私有 CBDB 数据构建朝代专属 Markov 模型。",
    compareTitle: "模型对比",
    compareSummary:
      "用同一组提示参数，让所有可用模型同时生成结果，并选择你最喜欢的一组。",
    favoritePick: "选这一组",
    favoriteSaved: "偏好已记录。",
    compareEmpty: "先执行一次模型对比，这里会展示并排结果。",
    blindPairwiseTitle: "匿名双选",
    blindPairwiseSummary:
      "两组匿名结果，一次选择。这是当前 public feedback 里最主要的轻量任务。",
    blindPrompt: "哪一组更像合理的中文名字？",
    dynastyGuessTitle: "朝代猜测",
    dynastyGuessSummary: "基于朝代条件 Markov 输出的一个小型风格辨识任务。",
    dynastyGuessPrompt: "这组名字整体更像哪个朝代？",
    batchA: "A 组",
    batchB: "B 组",
    nextTask: "加载下一题",
    refreshTask: "重新抽题",
    answers: {
      left: "选 A 组",
      right: "选 B 组",
      tie: "难分高下",
      skip: "跳过",
      Tang: "唐",
      Song: "宋",
      Ming: "明",
      Qing: "清",
    },
    feedbackAccepted: "反馈已记录，正在加载下一题...",
    feedbackUnavailable: "当前还没有可用的反馈存储。",
    loadingTask: "正在加载任务...",
    notesTitle: "架构说明",
    notesSummary: "这个公开页面不会隐藏较弱或较强的模型，这样研究和工程叙事才足够诚实。",
    notes: {
      markov: "适合展示可解释性、快速推理，以及很有辨识度的朝代玩法。",
      lstm: "在当前 fairness-first matched-pair 研究里仍然是表现最强的神经模型。",
      transformer: "作为真实部署的对照模型保留下来，而不是只停留在口头比较。",
    },
    github: "GitHub 仓库",
    blog: "技术 Blog",
    connectError: "连接 API 失败。",
    generationError: "生成失败。",
  },
} as const;

const generationToolsCopy = {
  en: {
    fixedToken: "Fixed token",
    fixedTokenPlaceholder: "明 / 若虚",
    tokenPosition: "Placement",
    lucky: "I'm feeling lucky",
    constraintHint:
      "Optional: lock one character or bigram in place and let the model fill the rest with constrained sampling.",
    matchMeta: "Constraint",
    supportMeta: "Corpus support",
    attemptsMeta: "Attempts",
    positions: {
      any: "Anywhere",
      start: "Start",
      middle: "Middle",
      end: "End",
    },
  },
  zh: {
    fixedToken: "\u6307\u5b9a\u5b57/\u8bcd",
    fixedTokenPlaceholder: "\u660e / \u82e5\u865a",
    tokenPosition: "\u4f4d\u7f6e",
    lucky: "\u8bd5\u8bd5\u624b\u6c14",
    constraintHint:
      "\u53ef\u9009\uff1a\u6307\u5b9a\u4e00\u4e2a\u5b57\u6216\u4e8c\u5b57\u8bcd\u7684\u4f4d\u7f6e\uff0c\u5176\u4f59\u90e8\u5206\u7531\u6a21\u578b\u7528\u7ea6\u675f\u91c7\u6837\u8865\u5168\u3002",
    matchMeta: "\u7ea6\u675f",
    supportMeta: "\u8bed\u6599\u652f\u6301\u5ea6",
    attemptsMeta: "\u91c7\u6837\u6b21\u6570",
    positions: {
      any: "\u4efb\u610f\u4f4d\u7f6e",
      start: "\u5f00\u5934",
      middle: "\u4e2d\u95f4",
      end: "\u7ed3\u5c3e",
    },
  },
} as const;

const workshopCopy = {
  en: {
    tab: "Name Workshop",
    title: "Name Workshop",
    summary:
      "For users who already have a favorite character or phrase, split the intent into historical pattern search and creative ancient-style completion.",
    historical: "Historical Pattern",
    creative: "Creative Ancient-Style",
    historicalIntro:
      "Evidence-based constrained search over the cleaned historical corpus. It answers whether the current data shows a usable naming pattern.",
    creativeIntro:
      "A template Fill-in-the-Middle route for creative completion. It is designed for controllability, not historical reconstruction.",
    fixedTokenRequired: "Enter a fixed token first.",
    supportLevel: "Evidence level",
    evidenceMessage: "Evidence note",
    satisfactionRate: "Constraint satisfaction",
    creativeUnavailable:
      "FIM model artifact is not available in this deployment. Historical Pattern mode is still usable.",
    creativeNotice:
      "Creative mode is not a historical authenticity claim; it is constrained ancient-style generation.",
    tryCreative: "Try Creative Ancient-Style Mode",
  },
  zh: {
    tab: "\u6307\u5b9a\u5b57\u53d6\u540d",
    title: "\u6307\u5b9a\u5b57\u53d6\u540d",
    summary:
      "\u5f53\u7528\u6237\u5df2\u7ecf\u6709\u559c\u6b22\u7684\u5b57\u6216\u4e8c\u5b57\u8bcd\u65f6\uff0c\u628a\u9700\u6c42\u62c6\u6210\u53f2\u6599\u6a21\u5f0f\u548c\u53e4\u98ce\u521b\u4f5c\u6a21\u5f0f\u3002",
    historical: "\u53f2\u6599\u6a21\u5f0f",
    creative: "\u53e4\u98ce\u521b\u4f5c\u6a21\u5f0f",
    historicalIntro:
      "\u57fa\u4e8e\u5f53\u524d\u6e05\u6d17\u540e\u7684\u5386\u53f2\u8bed\u6599\u505a\u7ea6\u675f\u641c\u7d22\uff0c\u56de\u7b54\u8bed\u6599\u91cc\u662f\u5426\u6709\u53ef\u89c2\u5bdf\u7684\u53d6\u540d\u6a21\u5f0f\u3002",
    creativeIntro:
      "\u9762\u5411\u521b\u4f5c\u7684 Fill-in-the-Middle \u8def\u7ebf\uff0c\u76ee\u6807\u662f\u66f4\u5f3a\u7684\u53ef\u63a7\u751f\u6210\uff0c\u4e0d\u4ee3\u8868\u5386\u53f2\u590d\u539f\u3002",
    fixedTokenRequired: "\u8bf7\u5148\u8f93\u5165\u6307\u5b9a\u5b57/\u8bcd\u3002",
    supportLevel: "\u8bc1\u636e\u7b49\u7ea7",
    evidenceMessage: "\u8bc1\u636e\u8bf4\u660e",
    satisfactionRate: "\u7ea6\u675f\u6ee1\u8db3\u7387",
    creativeUnavailable:
      "FIM \u6a21\u578b artifact \u5728\u5f53\u524d\u90e8\u7f72\u4e2d\u4e0d\u53ef\u7528\uff0c\u53f2\u6599\u6a21\u5f0f\u4ecd\u53ef\u4f7f\u7528\u3002",
    creativeNotice:
      "\u521b\u4f5c\u6a21\u5f0f\u4e0d\u662f\u5386\u53f2\u771f\u5b9e\u6027\u58f0\u660e\uff1b\u5b83\u662f\u53ef\u63a7\u7684\u53e4\u98ce\u751f\u6210\u5de5\u5177\u3002",
    tryCreative: "\u5207\u6362\u5230\u53e4\u98ce\u521b\u4f5c\u6a21\u5f0f",
  },
} as const;

function App() {
  const [locale, setLocale] = useState<Locale>(() => getInitialLocale());
  const [surface, setSurface] = useState<SurfaceMode>("playground");
  const [playgroundTab, setPlaygroundTab] = useState<"classic" | "dynasty" | "compare" | "workshop">("classic");
  const [evaluationTab, setEvaluationTab] = useState<EvalTaskType>("blind_pairwise");
  const [sessionId] = useState(() => getOrCreateSessionId());

  const [models, setModels] = useState<ModelStatus[]>([]);
  const [dynasties, setDynasties] = useState<DynastyOption[]>([]);
  const [bootError, setBootError] = useState<string | null>(null);
  const [isBootstrapping, setIsBootstrapping] = useState(true);

  const [selectedModel, setSelectedModel] = useState<ModelType>("lstm");
  const [seed, setSeed] = useState("");
  const [constraintText, setConstraintText] = useState("");
  const [constraintPosition, setConstraintPosition] = useState<ConstraintPosition>("any");
  const [count, setCount] = useState(5);
  const [temperature, setTemperature] = useState(0.8);
  const [classicResult, setClassicResult] = useState<GenerationResponse | null>(null);
  const [classicLoading, setClassicLoading] = useState(false);
  const [classicError, setClassicError] = useState<string | null>(null);

  const [selectedDynastyId, setSelectedDynastyId] = useState<number | null>(null);
  const [dynastySeed, setDynastySeed] = useState("");
  const [dynastyConstraintText, setDynastyConstraintText] = useState("");
  const [dynastyConstraintPosition, setDynastyConstraintPosition] = useState<ConstraintPosition>("any");
  const [dynastyCount, setDynastyCount] = useState(5);
  const [dynastyResult, setDynastyResult] = useState<GenerationResponse | null>(null);
  const [dynastyLoading, setDynastyLoading] = useState(false);
  const [dynastyError, setDynastyError] = useState<string | null>(null);

  const [compareSeed, setCompareSeed] = useState("");
  const [compareCount, setCompareCount] = useState(4);
  const [compareTemperature, setCompareTemperature] = useState(0.8);
  const [compareResult, setCompareResult] = useState<CompareResponse | null>(null);
  const [compareLoading, setCompareLoading] = useState(false);
  const [compareError, setCompareError] = useState<string | null>(null);
  const [compareFeedbackMessage, setCompareFeedbackMessage] = useState<string | null>(null);
  const [favoriteSubmittingModel, setFavoriteSubmittingModel] = useState<ModelType | null>(null);

  const [workshopMode, setWorkshopMode] = useState<WorkshopMode>("historical");
  const [workshopModel, setWorkshopModel] = useState<ModelType>("markov");
  const [workshopToken, setWorkshopToken] = useState("");
  const [workshopPosition, setWorkshopPosition] = useState<ConstraintPosition>("any");
  const [workshopSeed, setWorkshopSeed] = useState("");
  const [workshopCount, setWorkshopCount] = useState(5);
  const [workshopTemperature, setWorkshopTemperature] = useState(0.8);
  const [historicalResult, setHistoricalResult] = useState<HistoricalPatternResponse | null>(null);
  const [historicalLoading, setHistoricalLoading] = useState(false);
  const [historicalError, setHistoricalError] = useState<string | null>(null);
  const [infillResult, setInfillResult] = useState<InfillResponse | null>(null);
  const [infillLoading, setInfillLoading] = useState(false);
  const [infillError, setInfillError] = useState<string | null>(null);

  const [blindTask, setBlindTask] = useState<EvalTask | null>(null);
  const [blindLoadedAt, setBlindLoadedAt] = useState<number | null>(null);
  const [blindLoading, setBlindLoading] = useState(false);
  const [blindSubmitting, setBlindSubmitting] = useState(false);
  const [blindMessage, setBlindMessage] = useState<string | null>(null);
  const [blindError, setBlindError] = useState<string | null>(null);

  const [dynastyTask, setDynastyTask] = useState<EvalTask | null>(null);
  const [dynastyTaskLoadedAt, setDynastyTaskLoadedAt] = useState<number | null>(null);
  const [dynastyTaskLoading, setDynastyTaskLoading] = useState(false);
  const [dynastyTaskSubmitting, setDynastyTaskSubmitting] = useState(false);
  const [dynastyTaskMessage, setDynastyTaskMessage] = useState<string | null>(null);
  const [dynastyTaskError, setDynastyTaskError] = useState<string | null>(null);

  const text = copy[locale];
  const toolText = generationToolsCopy[locale];
  const workshopText = workshopCopy[locale];
  const currentModel = models.find((model) => model.model_type === selectedModel);
  const isMarkov = selectedModel === "markov";
  const currentWorkshopModel = models.find((model) => model.model_type === workshopModel);
  const isWorkshopMarkov = workshopModel === "markov";
  const availableDynasties = dynasties.filter((dynasty) => dynasty.available);
  const currentDynasty = dynasties.find((dynasty) => dynasty.dynasty_id === selectedDynastyId) ?? null;

  useEffect(() => {
    localStorage.setItem(LOCALE_STORAGE_KEY, locale);
    document.documentElement.lang = locale === "zh" ? "zh-CN" : "en";
  }, [locale]);

  useEffect(() => {
    async function bootstrap() {
      try {
        const [fetchedModels, fetchedDynasties] = await Promise.all([fetchModels(), fetchDynasties()]);
        setModels(fetchedModels);
        setDynasties(fetchedDynasties);

        const firstAvailableModel =
          modelOrder.find((modelType) =>
            fetchedModels.some((entry) => entry.model_type === modelType && entry.available),
          ) ?? "lstm";
        setSelectedModel(firstAvailableModel);
        setWorkshopModel(firstAvailableModel);

        const firstAvailableDynasty = fetchedDynasties.find((entry) => entry.available);
        setSelectedDynastyId(firstAvailableDynasty?.dynasty_id ?? fetchedDynasties[0]?.dynasty_id ?? null);
      } catch (error) {
        setBootError(error instanceof Error ? error.message : text.connectError);
      } finally {
        setIsBootstrapping(false);
      }
    }

    bootstrap();
  }, [text.connectError]);

  useEffect(() => {
    if (surface !== "evaluation_lab") {
      return;
    }

    if (evaluationTab === "blind_pairwise" && !blindTask && !blindLoading) {
      void loadEvalTask("blind_pairwise");
    }
    if (evaluationTab === "dynasty_guess" && !dynastyTask && !dynastyTaskLoading) {
      void loadEvalTask("dynasty_guess");
    }
  }, [surface, evaluationTab, blindTask, blindLoading, dynastyTask, dynastyTaskLoading]);

  function dynastyLabel(dynasty: DynastyOption): string {
    return locale === "zh" ? dynasty.label_zh : dynasty.label_en;
  }

  function constraintPositionLabel(position: ConstraintPosition): string {
    return toolText.positions[position];
  }

  async function requestClassicGeneration(options?: { lucky?: boolean }) {
    const lucky = options?.lucky ?? false;
    setClassicError(null);
    setClassicLoading(true);

    try {
      const activeSeed = lucky ? "" : seed.trim();
      const activeConstraintText = lucky ? "" : constraintText.trim();
      const response = await generateNames({
        model_type: selectedModel,
        count,
        ...(activeSeed ? { seed: activeSeed } : {}),
        ...(activeConstraintText
          ? {
              constraint_text: activeConstraintText,
              constraint_position: constraintPosition,
            }
          : {}),
        ...(!isMarkov ? { temperature } : {}),
      });
      setClassicResult(response);
      if (lucky) {
        setSeed("");
        setConstraintText("");
        setConstraintPosition("any");
      }
    } catch (error) {
      setClassicError(error instanceof Error ? error.message : text.generationError);
    } finally {
      setClassicLoading(false);
    }
  }

  async function handleClassicSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    await requestClassicGeneration();
  }

  async function handleClassicLucky() {
    await requestClassicGeneration({ lucky: true });
  }

  async function requestDynastyGeneration(options?: { lucky?: boolean }) {
    const lucky = options?.lucky ?? false;
    if (!selectedDynastyId) {
      return;
    }

    setDynastyError(null);
    setDynastyLoading(true);

    try {
      const activeSeed = lucky ? "" : dynastySeed.trim();
      const activeConstraintText = lucky ? "" : dynastyConstraintText.trim();
      const response = await generateDynastyNames({
        dynasty_id: selectedDynastyId,
        count: dynastyCount,
        ...(activeSeed ? { seed: activeSeed } : {}),
        ...(activeConstraintText
          ? {
              constraint_text: activeConstraintText,
              constraint_position: dynastyConstraintPosition,
            }
          : {}),
      });
      setDynastyResult(response);
      if (lucky) {
        setDynastySeed("");
        setDynastyConstraintText("");
        setDynastyConstraintPosition("any");
      }
    } catch (error) {
      setDynastyError(error instanceof Error ? error.message : text.generationError);
    } finally {
      setDynastyLoading(false);
    }
  }

  async function handleDynastySubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    await requestDynastyGeneration();
  }

  async function handleDynastyLucky() {
    await requestDynastyGeneration({ lucky: true });
  }

  async function handleCompareSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setCompareError(null);
    setCompareFeedbackMessage(null);
    setCompareLoading(true);

    try {
      const response = await compareModels({
        count: compareCount,
        ...(compareSeed.trim() ? { seed: compareSeed.trim() } : {}),
        temperature: compareTemperature,
      });
      setCompareResult(response);
    } catch (error) {
      setCompareError(error instanceof Error ? error.message : text.generationError);
    } finally {
      setCompareLoading(false);
    }
  }

  async function handleFavoritePick(modelType: ModelType) {
    if (!compareResult) {
      return;
    }

    setFavoriteSubmittingModel(modelType);
    setCompareFeedbackMessage(null);

    try {
      await submitFeedback({
        session_id: sessionId,
        locale,
        surface: "playground",
        task_type: "favorite_pick",
        run_id: compareResult.run_id,
        checkpoint_id: "compare_live",
        prompt_type: "live_compare",
        response_label: modelType,
        latency_ms: 0,
        response_payload: { selected_model: modelType },
        presented_payload: {
          request_metadata: compareResult.request_metadata,
          results: compareResult.results.map((item) => ({
            model_type: item.model_type,
            generations: item.generations,
            metadata: item.metadata,
          })),
        },
      });
      setCompareFeedbackMessage(text.favoriteSaved);
    } catch (error) {
      setCompareFeedbackMessage(error instanceof Error ? error.message : text.feedbackUnavailable);
    } finally {
      setFavoriteSubmittingModel(null);
    }
  }

  async function handleWorkshopSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const fixedToken = workshopToken.trim();
    if (!fixedToken) {
      if (workshopMode === "historical") {
        setHistoricalError(workshopText.fixedTokenRequired);
      } else {
        setInfillError(workshopText.fixedTokenRequired);
      }
      return;
    }

    if (workshopMode === "historical") {
      setHistoricalError(null);
      setHistoricalLoading(true);
      try {
        const response = await generateHistoricalPattern({
          model_type: workshopModel,
          fixed_token: fixedToken,
          position: workshopPosition,
          count: workshopCount,
          ...(workshopSeed.trim() ? { seed: workshopSeed.trim() } : {}),
          ...(!isWorkshopMarkov ? { temperature: workshopTemperature } : {}),
        });
        setHistoricalResult(response);
      } catch (error) {
        setHistoricalError(error instanceof Error ? error.message : text.generationError);
      } finally {
        setHistoricalLoading(false);
      }
      return;
    }

    setInfillError(null);
    setInfillLoading(true);
    try {
      const response = await generateInfill({
        fixed_token: fixedToken,
        position: workshopPosition,
        count: workshopCount,
        ...(workshopSeed.trim() ? { seed: workshopSeed.trim() } : {}),
        temperature: workshopTemperature,
      });
      setInfillResult(response);
    } catch (error) {
      setInfillError(error instanceof Error ? error.message : workshopText.creativeUnavailable);
    } finally {
      setInfillLoading(false);
    }
  }

  async function loadEvalTask(taskType: EvalTaskType) {
    if (taskType === "blind_pairwise") {
      setBlindLoading(true);
      setBlindError(null);
      setBlindMessage(null);
      try {
        const task = await fetchEvalTask(taskType);
        setBlindTask(task);
        setBlindLoadedAt(Date.now());
      } catch (error) {
        setBlindError(error instanceof Error ? error.message : text.connectError);
      } finally {
        setBlindLoading(false);
      }
      return;
    }

    setDynastyTaskLoading(true);
    setDynastyTaskError(null);
    setDynastyTaskMessage(null);
    try {
      const task = await fetchEvalTask(taskType);
      setDynastyTask(task);
      setDynastyTaskLoadedAt(Date.now());
    } catch (error) {
      setDynastyTaskError(error instanceof Error ? error.message : text.connectError);
    } finally {
      setDynastyTaskLoading(false);
    }
  }

  async function submitEvalFeedback(taskType: EvalTaskType, responseLabel: string) {
    if (taskType === "blind_pairwise" && blindTask) {
      setBlindSubmitting(true);
      setBlindError(null);
      try {
        await submitFeedback({
          session_id: sessionId,
          locale,
          surface: "evaluation_lab",
          task_type: "blind_pairwise",
          task_id: blindTask.task_id,
          response_label: responseLabel,
          response_payload: { selected: responseLabel },
          latency_ms: Math.max(0, Date.now() - (blindLoadedAt ?? Date.now())),
          presented_payload: blindTask.presented_payload as unknown as Record<string, unknown>,
        });
        setBlindMessage(text.feedbackAccepted);
        await loadEvalTask("blind_pairwise");
      } catch (error) {
        setBlindError(error instanceof Error ? error.message : text.feedbackUnavailable);
      } finally {
        setBlindSubmitting(false);
      }
      return;
    }

    if (taskType === "dynasty_guess" && dynastyTask) {
      setDynastyTaskSubmitting(true);
      setDynastyTaskError(null);
      try {
        await submitFeedback({
          session_id: sessionId,
          locale,
          surface: "evaluation_lab",
          task_type: "dynasty_guess",
          task_id: dynastyTask.task_id,
          response_label: responseLabel,
          response_payload: { selected: responseLabel },
          latency_ms: Math.max(0, Date.now() - (dynastyTaskLoadedAt ?? Date.now())),
          presented_payload: dynastyTask.presented_payload as unknown as Record<string, unknown>,
        });
        setDynastyTaskMessage(text.feedbackAccepted);
        await loadEvalTask("dynasty_guess");
      } catch (error) {
        setDynastyTaskError(error instanceof Error ? error.message : text.feedbackUnavailable);
      } finally {
        setDynastyTaskSubmitting(false);
      }
    }
  }

  function renderClassicTab() {
    return (
      <div className="content-stack">
        <div className="panel__header panel__header--centered">
          <h2>{text.classicTitle}</h2>
          <p>{text.classicSummary}</p>
        </div>

        <div className="model-grid">
          {modelOrder.map((modelType) => {
            const model = models.find((entry) => entry.model_type === modelType);
            if (!model) {
              return null;
            }
            return (
              <ModelCard
                key={model.model_type}
                locale={locale}
                model={model}
                selected={selectedModel === model.model_type}
                onSelect={setSelectedModel}
              />
            );
          })}
        </div>

        <form className="generator-form generator-form--wide" onSubmit={handleClassicSubmit}>
          <div className="generator-form__row">
            <label>
              {text.selectedModel}
              <div className="selection-chip">{selectedModel.toUpperCase()}</div>
            </label>

            <label>
              {text.seed}
              <input
                value={seed}
                onChange={(event) => setSeed(event.target.value)}
                placeholder={text.seedPlaceholder}
                maxLength={2}
              />
            </label>

            <label>
              {text.count}
              <input
                type="number"
                min={1}
                max={20}
                value={count}
                onChange={(event) => setCount(Number(event.target.value))}
              />
            </label>
          </div>

          <div className="generator-form__row">
            <label>
              {toolText.fixedToken}
              <input
                value={constraintText}
                onChange={(event) => setConstraintText(event.target.value)}
                placeholder={toolText.fixedTokenPlaceholder}
                maxLength={2}
              />
            </label>

            <label>
              {toolText.tokenPosition}
              <select
                value={constraintPosition}
                onChange={(event) => setConstraintPosition(event.target.value as ConstraintPosition)}
                disabled={!constraintText.trim()}
              >
                {constraintPositionOrder.map((position) => (
                  <option key={position} value={position}>
                    {constraintPositionLabel(position)}
                  </option>
                ))}
              </select>
            </label>

            <div className="message message--soft generator-note">{toolText.constraintHint}</div>
          </div>

          <div className="generator-form__row generator-form__row--actions">
            <label className={isMarkov ? "is-disabled" : ""}>
              {text.temperature}
              <input
                type="range"
                min={0.2}
                max={1.5}
                step={0.1}
                value={temperature}
                onChange={(event) => setTemperature(Number(event.target.value))}
                disabled={isMarkov}
              />
              <span className="range-label">
                {isMarkov ? text.markovTemperature : temperature.toFixed(1)}
              </span>
            </label>

            <div className="button-cluster">
              <button
                type="button"
                className="lucky-button"
                onClick={() => void handleClassicLucky()}
                disabled={classicLoading || !currentModel?.available}
              >
                {toolText.lucky}
              </button>
              <button type="submit" disabled={classicLoading || !currentModel?.available}>
                {classicLoading ? text.generating : text.generate}
              </button>
            </div>
          </div>
        </form>

        {classicError ? <div className="message message--error">{classicError}</div> : null}

        <section className="subpanel">
          <div className="panel__header panel__header--centered">
            <h3>{text.resultsTitle}</h3>
            <p>{text.resultsSummary}</p>
          </div>
          <ResultList emptyText={text.emptyState} items={classicResult?.generations ?? []} />
          {classicResult ? (
            <div className="metadata metadata--centered">
              <span>{text.source}: {classicResult.metadata.source}</span>
              <span>
                {text.seedMeta}: {classicResult.metadata.normalized_seed ?? text.none}
                {classicResult.metadata.seed_was_normalized ? text.normalizedSuffix : ""}
              </span>
              <span>{text.countMeta}: {classicResult.metadata.requested_count}</span>
              {classicResult.metadata.normalized_constraint_text ? (
                <span>
                  {toolText.matchMeta}: {classicResult.metadata.normalized_constraint_text} |{" "}
                  {constraintPositionLabel(
                    (classicResult.metadata.constraint_position ?? "any") as ConstraintPosition,
                  )}{" "}
                  | {classicResult.metadata.constraint_match_count}/{classicResult.metadata.requested_count}
                </span>
              ) : null}
              {classicResult.metadata.normalized_constraint_text ? (
                <span>
                  {toolText.supportMeta}: {classicResult.metadata.constraint_support_count ?? text.none}
                </span>
              ) : null}
              {classicResult.metadata.normalized_constraint_text ? (
                <span>
                  {toolText.attemptsMeta}: {classicResult.metadata.sampling_attempts ?? text.none}
                </span>
              ) : null}
            </div>
          ) : null}
          {classicResult?.metadata.warning ? (
            <div className="message message--soft">{classicResult.metadata.warning}</div>
          ) : null}
        </section>
      </div>
    );
  }

  function renderDynastyTab() {
    return (
      <div className="content-stack">
        <div className="panel__header panel__header--centered">
          <h2>{text.dynastyTitle}</h2>
          <p>{text.dynastySummary}</p>
        </div>

        {availableDynasties.length > 0 ? (
          <div className="pill-row">
            {dynasties.map((dynasty) => (
              <button
                key={dynasty.dynasty_id}
                type="button"
                className={`pill-button ${selectedDynastyId === dynasty.dynasty_id ? "is-active" : ""}`}
                onClick={() => setSelectedDynastyId(dynasty.dynasty_id)}
                disabled={!dynasty.available}
              >
                {dynastyLabel(dynasty)}
              </button>
            ))}
          </div>
        ) : (
          <div className="message">{text.dynastyUnavailable}</div>
        )}

        <div className="message message--soft">{text.dynastyHint}</div>

        <form className="generator-form generator-form--wide" onSubmit={handleDynastySubmit}>
          <div className="generator-form__row">
            <label>
              {text.dynastyLabel}
              <div className="selection-chip">
                {currentDynasty ? dynastyLabel(currentDynasty) : text.none}
              </div>
            </label>

            <label>
              {text.seed}
              <input
                value={dynastySeed}
                onChange={(event) => setDynastySeed(event.target.value)}
                placeholder={text.seedPlaceholder}
                maxLength={2}
              />
            </label>

            <label>
              {text.count}
              <input
                type="number"
                min={1}
                max={20}
                value={dynastyCount}
                onChange={(event) => setDynastyCount(Number(event.target.value))}
              />
            </label>
          </div>

          <div className="generator-form__row">
            <label>
              {toolText.fixedToken}
              <input
                value={dynastyConstraintText}
                onChange={(event) => setDynastyConstraintText(event.target.value)}
                placeholder={toolText.fixedTokenPlaceholder}
                maxLength={2}
              />
            </label>

            <label>
              {toolText.tokenPosition}
              <select
                value={dynastyConstraintPosition}
                onChange={(event) => setDynastyConstraintPosition(event.target.value as ConstraintPosition)}
                disabled={!dynastyConstraintText.trim()}
              >
                {constraintPositionOrder.map((position) => (
                  <option key={position} value={position}>
                    {constraintPositionLabel(position)}
                  </option>
                ))}
              </select>
            </label>

            <div className="message message--soft generator-note">{toolText.constraintHint}</div>
          </div>

          <div className="generator-form__row generator-form__row--actions">
            <div className="field-spacer" />
            <div className="button-cluster">
              <button
                type="button"
                className="lucky-button"
                onClick={() => void handleDynastyLucky()}
                disabled={dynastyLoading || !currentDynasty?.available}
              >
                {toolText.lucky}
              </button>
              <button type="submit" disabled={dynastyLoading || !currentDynasty?.available}>
                {dynastyLoading ? text.generating : text.generate}
              </button>
            </div>
          </div>
        </form>

        {dynastyError ? <div className="message message--error">{dynastyError}</div> : null}

        <section className="subpanel">
          <div className="panel__header panel__header--centered">
            <h3>{text.resultsTitle}</h3>
            <p>{text.resultsSummary}</p>
          </div>
          <ResultList emptyText={text.emptyState} items={dynastyResult?.generations ?? []} />
          {dynastyResult ? (
            <div className="metadata metadata--centered">
              <span>{text.source}: {dynastyResult.metadata.source}</span>
              <span>
                {text.dynastyLabel}:{" "}
                {locale === "zh"
                  ? dynastyResult.metadata.dynasty_label_zh
                  : dynastyResult.metadata.dynasty_label_en}
              </span>
              <span>
                {text.seedMeta}: {dynastyResult.metadata.normalized_seed ?? text.none}
                {dynastyResult.metadata.seed_was_normalized ? text.normalizedSuffix : ""}
              </span>
              {dynastyResult.metadata.normalized_constraint_text ? (
                <span>
                  {toolText.matchMeta}: {dynastyResult.metadata.normalized_constraint_text} |{" "}
                  {constraintPositionLabel(
                    (dynastyResult.metadata.constraint_position ?? "any") as ConstraintPosition,
                  )}{" "}
                  | {dynastyResult.metadata.constraint_match_count}/{dynastyResult.metadata.requested_count}
                </span>
              ) : null}
              {dynastyResult.metadata.normalized_constraint_text ? (
                <span>
                  {toolText.supportMeta}: {dynastyResult.metadata.constraint_support_count ?? text.none}
                </span>
              ) : null}
              {dynastyResult.metadata.normalized_constraint_text ? (
                <span>
                  {toolText.attemptsMeta}: {dynastyResult.metadata.sampling_attempts ?? text.none}
                </span>
              ) : null}
            </div>
          ) : null}
          {dynastyResult?.metadata.warning ? (
            <div className="message message--soft">{dynastyResult.metadata.warning}</div>
          ) : null}
        </section>
      </div>
    );
  }

  function renderCompareTab() {
    return (
      <div className="content-stack">
        <div className="panel__header panel__header--centered">
          <h2>{text.compareTitle}</h2>
          <p>{text.compareSummary}</p>
        </div>

        <form className="generator-form generator-form--wide" onSubmit={handleCompareSubmit}>
          <div className="generator-form__row">
            <label>
              {text.seed}
              <input
                value={compareSeed}
                onChange={(event) => setCompareSeed(event.target.value)}
                placeholder={text.seedPlaceholder}
                maxLength={2}
              />
            </label>

            <label>
              {text.count}
              <input
                type="number"
                min={1}
                max={20}
                value={compareCount}
                onChange={(event) => setCompareCount(Number(event.target.value))}
              />
            </label>

            <label>
              {text.temperature}
              <input
                type="range"
                min={0.2}
                max={1.5}
                step={0.1}
                value={compareTemperature}
                onChange={(event) => setCompareTemperature(Number(event.target.value))}
              />
              <span className="range-label">{compareTemperature.toFixed(1)}</span>
            </label>
          </div>

          <div className="generator-form__row generator-form__row--actions">
            <div className="field-spacer" />
            <button type="submit" disabled={compareLoading}>
              {compareLoading ? text.generating : text.generate}
            </button>
          </div>
        </form>

        {compareError ? <div className="message message--error">{compareError}</div> : null}
        {compareFeedbackMessage ? <div className="message">{compareFeedbackMessage}</div> : null}

        {compareResult ? (
          <div className="compare-grid">
            {compareResult.results.map((result) => (
              <article key={result.model_type} className="compare-card">
                <div className="compare-card__header">
                  <h3>{result.model_type.toUpperCase()}</h3>
                  <span className="compare-card__meta">{result.metadata.source}</span>
                </div>
                <ResultList emptyText={text.compareEmpty} items={result.generations} />
                <button
                  type="button"
                  className="secondary-button"
                  onClick={() => handleFavoritePick(result.model_type)}
                  disabled={favoriteSubmittingModel === result.model_type}
                >
                  {favoriteSubmittingModel === result.model_type ? text.generating : text.favoritePick}
                </button>
              </article>
            ))}
          </div>
        ) : (
          <div className="empty-state">
            <p>{text.compareEmpty}</p>
          </div>
        )}
      </div>
    );
  }

  function renderNameWorkshopTab() {
    const activeLoading = workshopMode === "historical" ? historicalLoading : infillLoading;
    const activeIntro =
      workshopMode === "historical" ? workshopText.historicalIntro : workshopText.creativeIntro;

    return (
      <div className="content-stack">
        <div className="panel__header panel__header--centered">
          <h2>{workshopText.title}</h2>
          <p>{workshopText.summary}</p>
        </div>

        <div className="mode-card-grid">
          {(["historical", "creative"] as const).map((mode) => (
            <button
              key={mode}
              type="button"
              className={`mode-card ${workshopMode === mode ? "is-active" : ""}`}
              onClick={() => setWorkshopMode(mode)}
            >
              <span>{mode === "historical" ? workshopText.historical : workshopText.creative}</span>
              <small>
                {mode === "historical" ? workshopText.historicalIntro : workshopText.creativeIntro}
              </small>
            </button>
          ))}
        </div>

        <div className="message message--soft">{activeIntro}</div>

        {workshopMode === "historical" ? (
          <div className="model-grid">
            {modelOrder.map((modelType) => {
              const model = models.find((entry) => entry.model_type === modelType);
              if (!model) {
                return null;
              }
              return (
                <ModelCard
                  key={model.model_type}
                  locale={locale}
                  model={model}
                  selected={workshopModel === model.model_type}
                  onSelect={setWorkshopModel}
                />
              );
            })}
          </div>
        ) : null}

        <form className="generator-form generator-form--wide" onSubmit={handleWorkshopSubmit}>
          <div className="generator-form__row">
            <label>
              {toolText.fixedToken}
              <input
                value={workshopToken}
                onChange={(event) => setWorkshopToken(event.target.value)}
                placeholder={toolText.fixedTokenPlaceholder}
                maxLength={2}
              />
            </label>

            <label>
              {toolText.tokenPosition}
              <select
                value={workshopPosition}
                onChange={(event) => setWorkshopPosition(event.target.value as ConstraintPosition)}
              >
                {constraintPositionOrder.map((position) => (
                  <option key={position} value={position}>
                    {constraintPositionLabel(position)}
                  </option>
                ))}
              </select>
            </label>

            <label>
              {text.count}
              <input
                type="number"
                min={1}
                max={10}
                value={workshopCount}
                onChange={(event) => setWorkshopCount(Number(event.target.value))}
              />
            </label>
          </div>

          <div className="generator-form__row generator-form__row--actions">
            <label>
              {text.seed}
              <input
                value={workshopSeed}
                onChange={(event) => setWorkshopSeed(event.target.value)}
                placeholder={text.seedPlaceholder}
                maxLength={2}
              />
            </label>

            <label
              className={workshopMode === "historical" && isWorkshopMarkov ? "is-disabled" : ""}
            >
              {text.temperature}
              <input
                type="range"
                min={0.2}
                max={1.5}
                step={0.1}
                value={workshopTemperature}
                onChange={(event) => setWorkshopTemperature(Number(event.target.value))}
                disabled={workshopMode === "historical" && isWorkshopMarkov}
              />
              <span className="range-label">
                {workshopMode === "historical" && isWorkshopMarkov
                  ? text.markovTemperature
                  : workshopTemperature.toFixed(1)}
              </span>
            </label>

            <button
              type="submit"
              disabled={activeLoading || (workshopMode === "historical" && !currentWorkshopModel?.available)}
            >
              {activeLoading ? text.generating : text.generate}
            </button>
          </div>
        </form>

        {historicalError && workshopMode === "historical" ? (
          <div className="message message--error">{historicalError}</div>
        ) : null}
        {infillError && workshopMode === "creative" ? (
          <div className="message message--error">{infillError}</div>
        ) : null}

        {workshopMode === "historical" ? (
          <section className="subpanel">
            <div className="panel__header panel__header--centered">
              <h3>{text.resultsTitle}</h3>
              <p>{workshopText.historicalIntro}</p>
            </div>
            <ResultList emptyText={text.emptyState} items={historicalResult?.generations ?? []} />
            {historicalResult ? (
              <>
                <div className="metadata metadata--centered">
                  <span>{toolText.matchMeta}: {historicalResult.normalized_fixed_token}</span>
                  <span>{toolText.supportMeta}: {historicalResult.support_count}</span>
                  <span>{workshopText.supportLevel}: {historicalResult.support_level}</span>
                  <span>{toolText.attemptsMeta}: {historicalResult.sampling_attempts}</span>
                </div>
                <div className="message message--soft">
                  {workshopText.evidenceMessage}: {historicalResult.evidence_message}
                </div>
                {historicalResult.support_level === "none" || historicalResult.search_exhausted ? (
                  <button
                    type="button"
                    className="secondary-button workshop-cta"
                    onClick={() => setWorkshopMode("creative")}
                  >
                    {workshopText.tryCreative}
                  </button>
                ) : null}
              </>
            ) : null}
          </section>
        ) : (
          <section className="subpanel">
            <div className="panel__header panel__header--centered">
              <h3>{text.resultsTitle}</h3>
              <p>{workshopText.creativeIntro}</p>
            </div>
            <ResultList emptyText={text.emptyState} items={infillResult?.generations ?? []} />
            {infillResult ? (
              <>
                <div className="metadata metadata--centered">
                  <span>{toolText.matchMeta}: {infillResult.normalized_fixed_token}</span>
                  <span>
                    {workshopText.satisfactionRate}:{" "}
                    {(infillResult.constraint_satisfaction_rate * 100).toFixed(0)}%
                  </span>
                  <span>{workshopText.supportLevel}: {infillResult.historical_support_level}</span>
                </div>
                <div className="message message--soft">
                  {infillResult.creative_mode_notice || workshopText.creativeNotice}
                </div>
              </>
            ) : (
              <div className="message message--soft">{workshopText.creativeNotice}</div>
            )}
          </section>
        )}
      </div>
    );
  }

  function renderBlindPairwiseTask() {
    const payload = blindTask && isBlindPairwisePayload(blindTask.presented_payload)
      ? blindTask.presented_payload
      : null;

    return (
      <div className="content-stack">
        <div className="panel__header panel__header--centered">
          <h2>{text.blindPairwiseTitle}</h2>
          <p>{text.blindPairwiseSummary}</p>
        </div>

        {blindLoading ? <div className="message">{text.loadingTask}</div> : null}
        {blindError ? <div className="message message--error">{blindError}</div> : null}
        {blindMessage ? <div className="message">{blindMessage}</div> : null}

        {payload ? (
          <div className="task-card">
            <p className="task-card__prompt">{blindTask?.prompt ?? text.blindPrompt}</p>
            <div className="task-columns">
              <section className="task-column">
                <h3>{text.batchA}</h3>
                <ResultList emptyText={text.emptyState} items={payload.left.names} />
              </section>
              <section className="task-column">
                <h3>{text.batchB}</h3>
                <ResultList emptyText={text.emptyState} items={payload.right.names} />
              </section>
            </div>
            <div className="answer-grid">
              {(["left", "right", "tie", "skip"] as const).map((answer) => (
                <button
                  key={answer}
                  type="button"
                  className="secondary-button"
                  onClick={() => submitEvalFeedback("blind_pairwise", answer)}
                  disabled={blindSubmitting}
                >
                  {text.answers[answer]}
                </button>
              ))}
            </div>
            <button
              type="button"
              className="ghost-button"
              onClick={() => loadEvalTask("blind_pairwise")}
              disabled={blindLoading || blindSubmitting}
            >
              {text.refreshTask}
            </button>
          </div>
        ) : null}
      </div>
    );
  }

  function renderDynastyGuessTask() {
    const payload = dynastyTask && isDynastyGuessPayload(dynastyTask.presented_payload)
      ? dynastyTask.presented_payload
      : null;

    return (
      <div className="content-stack">
        <div className="panel__header panel__header--centered">
          <h2>{text.dynastyGuessTitle}</h2>
          <p>{text.dynastyGuessSummary}</p>
        </div>

        {dynastyTaskLoading ? <div className="message">{text.loadingTask}</div> : null}
        {dynastyTaskError ? <div className="message message--error">{dynastyTaskError}</div> : null}
        {dynastyTaskMessage ? <div className="message">{dynastyTaskMessage}</div> : null}

        {payload ? (
          <div className="task-card task-card--narrow">
            <p className="task-card__prompt">{dynastyTask?.prompt ?? text.dynastyGuessPrompt}</p>
            <ResultList emptyText={text.emptyState} items={payload.names} />
            <div className="answer-grid answer-grid--dynasty">
              {(["Tang", "Song", "Ming", "Qing", "skip"] as const).map((answer) => (
                <button
                  key={answer}
                  type="button"
                  className="secondary-button"
                  onClick={() => submitEvalFeedback("dynasty_guess", answer)}
                  disabled={dynastyTaskSubmitting}
                >
                  {text.answers[answer]}
                </button>
              ))}
            </div>
            <button
              type="button"
              className="ghost-button"
              onClick={() => loadEvalTask("dynasty_guess")}
              disabled={dynastyTaskLoading || dynastyTaskSubmitting}
            >
              {text.refreshTask}
            </button>
          </div>
        ) : null}
      </div>
    );
  }

  return (
    <div className="page-shell">
      <header className="hero hero--centered">
        <div className="hero__copy hero__copy--centered">
          <div className="hero__toolbar">
            <div className="toolbar-cluster">
              <span className="hero__toolbar-label">{text.surfaceLabel}</span>
              <div className="surface-toggle" role="group" aria-label={text.surfaceLabel}>
                <button
                  type="button"
                  className={`surface-toggle__button ${surface === "playground" ? "is-active" : ""}`}
                  onClick={() => setSurface("playground")}
                >
                  {text.playground}
                </button>
                <button
                  type="button"
                  className={`surface-toggle__button ${surface === "evaluation_lab" ? "is-active" : ""}`}
                  onClick={() => setSurface("evaluation_lab")}
                >
                  {text.evaluationLab}
                </button>
              </div>
            </div>

            <div className="toolbar-cluster">
              <span className="hero__toolbar-label">{text.languageLabel}</span>
              <div className="language-toggle" role="group" aria-label={text.languageLabel}>
                <button
                  type="button"
                  className={`language-toggle__button ${locale === "en" ? "is-active" : ""}`}
                  onClick={() => setLocale("en")}
                >
                  EN
                </button>
                <button
                  type="button"
                  className={`language-toggle__button ${locale === "zh" ? "is-active" : ""}`}
                  onClick={() => setLocale("zh")}
                >
                  中文
                </button>
              </div>
            </div>
          </div>
          <p className="eyebrow">{text.eyebrow}</p>
          <h1>{text.title}</h1>
          <p className="hero__summary">{text.heroSummary}</p>
        </div>
      </header>

      <main className="stack-layout">
        <section className="panel panel--wide panel--accent panel--centered">
          <div className="panel__header panel__header--centered">
            <h2>{surface === "playground" ? text.playground : text.evaluationLab}</h2>
            <p>{surface === "playground" ? text.playgroundSummary : text.evaluationSummary}</p>
          </div>

          {bootError ? <div className="message message--error">{bootError}</div> : null}
          {isBootstrapping ? <div className="message">{text.loadingTask}</div> : null}

          {!isBootstrapping ? (
            <>
              {surface === "playground" ? (
                <>
                  <div className="tab-row">
                    {(["classic", "dynasty", "compare", "workshop"] as const).map((tab) => (
                      <button
                        key={tab}
                        type="button"
                        className={`tab-button ${playgroundTab === tab ? "is-active" : ""}`}
                        onClick={() => setPlaygroundTab(tab)}
                      >
                        {tab === "workshop" ? workshopText.tab : text.tabs[tab]}
                      </button>
                    ))}
                  </div>
                  {playgroundTab === "classic" ? renderClassicTab() : null}
                  {playgroundTab === "dynasty" ? renderDynastyTab() : null}
                  {playgroundTab === "compare" ? renderCompareTab() : null}
                  {playgroundTab === "workshop" ? renderNameWorkshopTab() : null}
                </>
              ) : (
                <>
                  <div className="tab-row">
                    {(["blind_pairwise", "dynasty_guess"] as const).map((tab) => (
                      <button
                        key={tab}
                        type="button"
                        className={`tab-button ${evaluationTab === tab ? "is-active" : ""}`}
                        onClick={() => setEvaluationTab(tab)}
                      >
                        {text.tabs[tab === "blind_pairwise" ? "blindPairwise" : "dynastyGuess"]}
                      </button>
                    ))}
                  </div>
                  {evaluationTab === "blind_pairwise" ? renderBlindPairwiseTask() : null}
                  {evaluationTab === "dynasty_guess" ? renderDynastyGuessTask() : null}
                </>
              )}
            </>
          ) : null}
        </section>

        <section className="panel panel--wide panel--centered">
          <div className="panel__header panel__header--centered">
            <h2>{text.notesTitle}</h2>
            <p>{text.notesSummary}</p>
          </div>
          <div className="notes-grid notes-grid--centered">
            <article>
              <h3>Markov</h3>
              <p>{text.notes.markov}</p>
            </article>
            <article>
              <h3>LSTM</h3>
              <p>{text.notes.lstm}</p>
            </article>
            <article>
              <h3>Transformer</h3>
              <p>{text.notes.transformer}</p>
            </article>
          </div>
          <div className="link-row link-row--centered">
            <a href={REPO_URL} target="_blank" rel="noreferrer">
              {text.github}
            </a>
            <a href={BLOG_URL} target="_blank" rel="noreferrer">
              {text.blog}
            </a>
          </div>
        </section>
      </main>
    </div>
  );
}

export default App;
