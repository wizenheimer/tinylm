/**
 * TinyLM - Lightweight language model inference with WebGPU acceleration
 */

// Export main TinyLM class
export { TinyLM } from './TinyLM';

// Export modules
export { GenerationModule } from './GenerationModule';
export { EmbeddingsModule } from './EmbeddingsModule';

// Export utilities
export { tryGarbageCollection, detectEnvironment } from './utils';
export type { EnvironmentInfo } from './utils';
export { ProgressTracker } from './ProgressTracker';
export { FileProgressTracker } from './FileProgressTracker';
export { WebGPUChecker } from './WebGPUChecker';
export { GenerationController, InterruptableStoppingCriteria } from './GenerationController';

// Export type definitions
export type {
  // General types
  ModelRegistryEntry,
  TokenizedInputs,
  WebGPUCapabilities,
  DeviceConfig,
  ProgressUpdate,
  CapabilityInfo,
  FileInfo,
  OverallProgress,

  // Generation types
  ChatMessage,
  CompletionOptions,
  CompletionResult,
  CompletionChunk,

  // Embedding types
  EmbeddingCreateOptions,
  EmbeddingResult,

  // Model types
  ModelLoadOptions,
  InitOptions,
  TinyLMOptions
} from './types';
