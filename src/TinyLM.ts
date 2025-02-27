/**
 * TinyLM - Modular library for text generation and embeddings
 */

import { tryGarbageCollection } from './utils';
import { ProgressTracker } from './ProgressTracker';
import { WebGPUChecker } from './WebGPUChecker';
import { GenerationModule } from './GenerationModule';
import { EmbeddingsModule } from './EmbeddingsModule';

import {
  TinyLMOptions,
  InitOptions,
  CapabilityInfo,
  CompletionOptions,
  CompletionResult,
  CompletionChunk,
  ModelLoadOptions,
  EmbeddingCreateOptions,
  EmbeddingResult
} from './types';

/**
 * Main TinyLM class that provides the OpenAI-compatible API
 */
export class TinyLM {
  private options: Required<TinyLMOptions>;
  private progressTracker: ProgressTracker;
  private webgpuChecker: WebGPUChecker;
  private initialized: boolean = false;

  // Modules
  private generationModule: GenerationModule;
  private embeddingsModule: EmbeddingsModule;

  // API structure
  readonly chat: {
    completions: {
      create: (options: CompletionOptions) => Promise<CompletionResult | AsyncGenerator<CompletionChunk>>;
    };
  };

  // Embeddings API
  readonly embeddings: {
    create: (options: EmbeddingCreateOptions) => Promise<EmbeddingResult>;
  };

  readonly models: {
    load: (options: ModelLoadOptions) => Promise<any>;
    offload: (options: { model: string }) => Promise<boolean>;
    interrupt: () => boolean;
    reset: () => void;
    check: () => Promise<CapabilityInfo>;
    list: () => string[];
  };

  /**
   * Create a new TinyLM instance
   * @param {TinyLMOptions} options - Configuration options
   */
  constructor(options: TinyLMOptions = {}) {
    this.options = {
      progressCallback: options.progressCallback || (() => {}),
      progressThrottleTime: options.progressThrottleTime || 100,
      ...options
    } as Required<TinyLMOptions>;

    // Initialize components
    this.progressTracker = new ProgressTracker(
      this.options.progressCallback,
      { throttleTime: this.options.progressThrottleTime }
    );
    this.webgpuChecker = new WebGPUChecker();

    // Initialize modules
    this.generationModule = new GenerationModule(this);
    this.embeddingsModule = new EmbeddingsModule(this);

    // Create API structure similar to OpenAI
    this.chat = {
      completions: {
        create: this.generationModule.createCompletion.bind(this.generationModule)
      }
    };

    // Embeddings API
    this.embeddings = {
      create: this.embeddingsModule.create.bind(this.embeddingsModule)
    };

    // Model management API
    this.models = {
      load: this.generationModule.loadModel.bind(this.generationModule),
      offload: this.generationModule.offloadModel.bind(this.generationModule),
      interrupt: this.generationModule.interrupt.bind(this.generationModule),
      reset: this.generationModule.reset.bind(this.generationModule),
      check: this.checkCapabilities.bind(this),
      list: () => Array.from(this.generationModule.getModelRegistry().keys())
    };
  }

  /**
   * Check hardware capabilities
   * @returns {Promise<CapabilityInfo>} Hardware capabilities
   */
  async checkCapabilities(): Promise<CapabilityInfo> {
    const gpuCapabilities = await this.webgpuChecker.check();

    // Get CPU/GPU info from environment when available
    let envInfo: Record<string, any> = {};
    try {
      // This would be environment-specific information from your original code
      envInfo = {
        // Include environment info here if available
        backend: 'unknown',
        cpuInfo: 'unknown',
        gpuInfo: 'unknown'
      };
    } catch (error) {
      // Environment variables not available
    }

    return {
      ...gpuCapabilities,
      environment: envInfo,
      transformersVersion: 'unknown' // This would be provided by your env variable
    };
  }

  /**
   * Initialize TinyLM with models
   * @param {InitOptions} options - Initialization options
   * @returns {Promise<TinyLM>} This instance
   */
  async init(options: InitOptions = {}): Promise<TinyLM> {
    if (!this.initialized) {
      // Check hardware capabilities
      const capabilities = await this.checkCapabilities();
      this.progressTracker.update({
        status: 'init',
        type: 'system',
        message: `Hardware check: WebGPU ${capabilities.isWebGPUSupported ? 'available' : 'not available'}`
      });

      // Initialize modules
      await this.generationModule.init({
        models: options.models || [],
        lazyLoad: options.lazyLoad
      });

      await this.embeddingsModule.init({
        defaultModel: options.embeddingModels && options.embeddingModels.length > 0
          ? options.embeddingModels[0]
          : undefined
      });

      this.initialized = true;
    }
    return this;
  }

  /**
   * Get the progress tracker instance (for module access)
   * @returns {ProgressTracker} Progress tracker
   */
  getProgressTracker(): ProgressTracker {
    return this.progressTracker;
  }

  /**
   * Get the WebGPU checker instance (for module access)
   * @returns {WebGPUChecker} WebGPU checker
   */
  getWebGPUChecker(): WebGPUChecker {
    return this.webgpuChecker;
  }
}
