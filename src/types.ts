/**
 * Common types for TinyLM
 */

import {
  PreTrainedTokenizer,
  PreTrainedModel,
} from "@huggingface/transformers";

/**
 * Interface for model output
 */
export interface ModelOutput {
  sequences?: number[][];
  past_key_values?: any;
  [key: string]: any;
}

/**
 * Interface for generate output
 */
export interface GenerateOutput extends ModelOutput {
  sequences: number[][];
  past_key_values?: any;
}

/**
 * Interface for stopping criteria list
 */
export interface StoppingCriteriaList {
  criteria: any[];
  push: (criterion: any) => void;
  extend: (criteria: any[]) => void;
  [Symbol.iterator](): IterableIterator<any>;
  shouldStop: (input_ids: number[][], scores: number[][], options?: any) => boolean;
}

/**
 * Interface for generation function parameters
 */
export interface GenerationFunctionParameters {
  input_ids?: number[][];
  attention_mask?: number[][];
  max_new_tokens?: number;
  temperature?: number;
  top_k?: number;
  top_p?: number;
  repetition_penalty?: number;
  stopping_criteria?: any;
  streamer?: any;
  return_dict_in_generate?: boolean;
  past_key_values?: any;
  [key: string]: any; // Allow additional properties
}

/**
 * WebGPU feature detection
 * Checks if WebGPU is supported and which features are available
 */
export interface WebGPUCapabilities {
  isWebGPUSupported: boolean;
  fp16Supported: boolean;
  isNode?: boolean;
  reason?: string;
  adapterInfo?: {
    name: string;
    description: string;
    features: string[];
    limits: Record<string, any>;
  };
}

/**
 * Device configuration for model loading
 */
export interface DeviceConfig {
  device?: string;
  dtype?: string;
}

/**
 * Interface for file information used by FileProgressTracker
 */
export interface FileInfo {
  id: string;
  name: string;
  status: string;
  progress: number;
  percentComplete: number;
  bytesLoaded: number;
  bytesTotal: number;
  startTime: number;
  lastUpdateTime: number;
  speed: number; // bytes per second
  timeRemaining: number | null; // in seconds
  // For compatibility with JS version:
  total?: number;
  [key: string]: any;
}

/**
 * Interface for overall progress information
 */
export interface OverallProgress {
  progress: number;
  percentComplete: number;
  bytesLoaded: number;
  bytesTotal: number;
  activeFileCount: number;
  totalFileCount: number;
  speed: number;
  timeRemaining: number;
  formattedLoaded: string;
  formattedTotal: string;
  formattedSpeed: string;
  formattedRemaining: string;
  isComplete: boolean;
  hasError: boolean;
}

/**
 * Interface for progress update
 */
export interface ProgressUpdate {
  status?: string;
  progress?: number;
  percentComplete?: number;
  message?: string;
  type?: string;
  [key: string]: any;
}

/**
 * Interface for progress tracker options
 */
export interface ProgressTrackerOptions {
  throttleTime?: number;
  significantChangeThreshold?: number;
  [key: string]: any;
}

/**
 * Interface for environment information
 */
export interface EnvInfo {
  backend?: string;
  cpuInfo?: string;
  gpuInfo?: string;
  [key: string]: any;
}

/**
 * Interface for capability information
 */
export interface CapabilityInfo extends WebGPUCapabilities {
  environment: EnvInfo;
  transformersVersion?: string;
}

/**
 * Model registry entry for generation models
 */
export interface ModelRegistryEntry {
  tokenizer: any;
  model: any;
}

/**
 * Information about a loaded embedding model
 */
export interface EmbeddingModelInfo {
  tokenizer: any;
  model: any;
  pipeline?: any;
  dimensions: number;
}

/**
 * Interface for the tokenized inputs
 */
export interface TokenizedInputs {
  input_ids: number[][];
  attention_mask?: number[][];
  [key: string]: any;
}

/**
 * Interface for a chat message
 */
export interface ChatMessage {
  role: string;
  content: string;
  [key: string]: any;
}

/**
 * Interface for chat completion options
 */
export interface CompletionOptions {
  messages: ChatMessage[];
  stream?: boolean;
  streamOptions?: any;
  model?: string | null;
  temperature?: number;
  max_tokens?: number;
  do_sample?: boolean;
  top_k?: number;
  top_p?: number;
  [key: string]: any;
}

/**
 * Interface for a completion result
 */
export interface CompletionResult {
  id: string;
  object: string;
  created: number;
  model: string;
  choices: Array<{
    index: number;
    message: ChatMessage;
    finish_reason: string;
  }>;
  _tinylm?: {
    time_ms: number;
  };
}

/**
 * Interface for a streaming completion chunk
 */
export interface CompletionChunk {
  id: string;
  object: string;
  created: number;
  model: string;
  choices: Array<{
    index: number;
    delta: {
      content?: string;
      [key: string]: any;
    };
    finish_reason: string | null;
  }>;
}

/**
 * Interface for TinyLM constructor options
 */
export interface TinyLMOptions {
  onProgress?: (progress: number) => void;
  progressThrottleTime?: number;
  [key: string]: any;
}

/**
 * Update InitOptions to include ttsModels
 */
export interface InitOptions {
  models?: string[];
  embeddingModels?: string[];
  ttsModels?: string[];
  lazyLoad?: boolean;
  [key: string]: any;
}

/**
 * Interface for embeddings creation options
 */
export interface EmbeddingCreateOptions {
  model: string;
  input: string | string[];
  encoding_format?: 'float' | 'base64';
  user?: string;
  dimensions?: number;
}

/**
 * Interface for embeddings result
 */
export interface EmbeddingResult {
  object: string;
  data: Array<{
    object: string;
    embedding: number[] | string; // number[] for float, string for base64
    index: number;
  }>;
  model: string;
  usage: {
    prompt_tokens: number;
    total_tokens: number;
  };
  _tinylm?: {
    time_ms: number;
    dimensions: number;
  };
}

/**
 * Interface for speech creation options with streaming
 */
export interface SpeechCreateOptions {
  model?: string;
  input: string;
  voice?: string;
  response_format?: 'mp3' | 'wav';
  speed?: number;
  stream?: boolean; // New parameter for streaming
  [key: string]: any;
}

/**
 * Interface for speech result
 */
export interface SpeechResult {
  id: string;
  object: string;
  created: number;
  model: string;
  audio: ArrayBuffer;
  content_type: string;
  _tinylm?: {
    time_ms: number;
  };
}

/**
 * Interface for streaming speech result
 */
export interface SpeechStreamResult {
  id: string;
  object: string;
  created: number;
  model: string;
  chunks: Array<{
    text: string;
    audio: ArrayBuffer;
    content_type: string;
  }>;
  _tinylm?: {
    time_ms: number;
  };
}

/**
 * Model types supported by TinyLM
 */
export enum ModelType {
  Generation = 'generation',
  Embedding = 'embedding',
  Audio = 'audio'
}

/**
 * Base interface for all model load options
 */
export interface BaseModelLoadOptions {
  model: string;
  type?: ModelType;
}

/**
 * Generation model load options
 */
export interface GenerationModelLoadOptions extends BaseModelLoadOptions {
  type: ModelType.Generation;
  quantization?: string;
}

/**
 * Embedding model load options
 */
export interface EmbeddingModelLoadOptions extends BaseModelLoadOptions {
  type: ModelType.Embedding;
  dimensions?: number;
}

/**
 * Audio model load options
 */
export interface AudioModelLoadOptions extends BaseModelLoadOptions {
  type: ModelType.Audio;
  device?: string;
  dtype?: string;
}

/**
 * Union type for all model load options
 */
export type ModelLoadOptions = GenerationModelLoadOptions | EmbeddingModelLoadOptions | AudioModelLoadOptions;

/**
 * Base interface for model registry entries
 */
export interface BaseModelInfo {
  model: any;
  tokenizer?: any;
}

/**
 * Generation model registry entry
 */
export interface GenerationModelInfo extends BaseModelInfo {
  tokenizer: any;
  model: any;
}

/**
 * Embedding model registry entry
 */
export interface EmbeddingModelInfo extends BaseModelInfo {
  tokenizer: any;
  model: any;
  pipeline?: any;
  dimensions: number;
}

/**
 * Audio model registry entry
 */
export interface AudioModelInfo extends BaseModelInfo {
  loaded: boolean;
}

export interface GenerationResult {
  text: string;
  tokens: number;
  timeMs: number;
}

export interface AudioResult {
  audio: ArrayBuffer;
  timeMs: number;
}

export interface ProgressTracker {
  onProgress(progress: number): void;
  throttleTime?: number;
}
