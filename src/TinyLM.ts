/**
 * TinyLM - Modular library for text generation and embeddings
 */

import { tryGarbageCollection, detectEnvironment, EnvironmentInfo } from './utils';
import { ProgressTracker } from './ProgressTracker';
import { WebGPUChecker } from './WebGPUChecker';
import { GenerationModule } from './GenerationModule';
import { EmbeddingsModule } from './EmbeddingsModule';
import { AudioModule } from './AudioModule';
import { ModelManager } from './ModelManager';
import { FileProgressTracker } from './FileProgressTracker';

import {
  TinyLMOptions,
  InitOptions,
  CapabilityInfo,
  CompletionOptions,
  CompletionResult,
  CompletionChunk,
  ModelLoadOptions,
  EmbeddingCreateOptions,
  EmbeddingResult,
  SpeechCreateOptions,
  SpeechResult,
  SpeechStreamResult,
  ProgressUpdate,
  ModelType,
  BaseModelLoadOptions,
  GenerationModelLoadOptions,
  EmbeddingModelLoadOptions,
  AudioModelLoadOptions,
  GenerationResult,
  AudioResult
} from './types';

/**
 * Main TinyLM class that provides the OpenAI-compatible API
 */
export class TinyLM {
  private options: TinyLMOptions;
  private progressTracker: ProgressTracker;
  private webgpuChecker: WebGPUChecker;
  private initialized: boolean = false;
  private environment: EnvironmentInfo;
  private modelManager: ModelManager;

  // Modules
  private generationModule: GenerationModule;
  private embeddingsModule: EmbeddingsModule;
  private audioModule: AudioModule;

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

  readonly audio: {
    speech: {
      create: (options: SpeechCreateOptions) => Promise<SpeechResult | SpeechStreamResult>;
    };
  };

  readonly models: {
    load: (options: BaseModelLoadOptions) => Promise<any>;
    offload: (options: BaseModelLoadOptions) => Promise<boolean>;
    interrupt: () => boolean;
    reset: () => void;
    check: () => Promise<CapabilityInfo>;
    list: (type?: ModelType) => string[];
  };

  /**
   * Create a new TinyLM instance
   * @param {TinyLMOptions} options - Configuration options
   */
  constructor(options: TinyLMOptions = {}) {
    this.options = {
      debug: options.debug ?? false,
      lazyLoad: options.lazyLoad ?? false,
      progressCallback: options.progressCallback ?? ((progress: ProgressUpdate) => { }),
      progressThrottleTime: options.progressThrottleTime ?? 100
    };

    // Detect the current environment
    this.environment = detectEnvironment();

    // Initialize core components
    this.webgpuChecker = new WebGPUChecker();
    this.progressTracker = new ProgressTracker(
      options.progressCallback,
      { throttleTime: options.progressThrottleTime }
    );

    this.modelManager = new ModelManager({
      webgpuChecker: this.webgpuChecker,
      progressTracker: this.progressTracker
    });

    // Initialize modules
    this.generationModule = new GenerationModule({ modelManager: this.modelManager });
    this.embeddingsModule = new EmbeddingsModule({ modelManager: this.modelManager });
    this.audioModule = new AudioModule({ modelManager: this.modelManager });

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

    // Audio API
    this.audio = {
      speech: {
        create: this.audioModule.createSpeech.bind(this.audioModule)
      }
    };

    // Model management API
    this.models = {
      load: async (options: BaseModelLoadOptions) => {
        return this.modelManager.loadModel(options);
      },
      offload: async (options: BaseModelLoadOptions) => {
        return this.modelManager.offloadModel(options);
      },
      interrupt: this.generationModule.interrupt.bind(this.generationModule),
      reset: this.generationModule.reset.bind(this.generationModule),
      check: this.checkCapabilities.bind(this),
      list: (type?: ModelType) => {
        return this.modelManager.getLoadedModels(type);
      }
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

      await this.audioModule.init({
        ttsModels: options.ttsModels || [],
        lazyLoad: options.lazyLoad
      });

      this.initialized = true;
    }
    return this;
  }

  /**
   * Check hardware capabilities
   * @returns {Promise<CapabilityInfo>} Hardware capabilities
   */
  async checkCapabilities(): Promise<CapabilityInfo> {
    const gpuCapabilities = await this.webgpuChecker.check();

    // Get runtime environment info
    const runtimeInfo = {
      ...this.environment,
      userAgent: typeof navigator !== 'undefined' ? navigator.userAgent : 'unknown'
    };

    // Get CPU/GPU info from environment when available
    let envInfo: Record<string, any> = {
      runtime: runtimeInfo,
      backend: 'unknown',
      cpuInfo: 'unknown',
      gpuInfo: 'unknown'
    };

    // In Node.js environments, try to get more detailed info
    if (this.environment.isNode) {
      try {
        // Only attempt to require these in Node environments
        const os = require('os');
        envInfo.cpuInfo = os.cpus()[0]?.model || 'unknown';
        envInfo.totalMemory = os.totalmem();
        envInfo.freeMemory = os.freemem();
        envInfo.platform = os.platform();
        envInfo.arch = os.arch();
      } catch (error) {
        // Ignore errors if modules can't be loaded
      }
    }

    return {
      ...gpuCapabilities,
      environment: envInfo,
      transformersVersion: 'unknown' // This would be provided by your env variable
    };
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

  /**
   * Get the model manager instance
   * @returns {ModelManager} Model manager
   */
  getModelManager(): ModelManager {
    return this.modelManager;
  }

  /**
   * Get the current environment information
   * @returns {EnvironmentInfo} Environment information
   */
  getEnvironment(): EnvironmentInfo {
    return this.environment;
  }
}
