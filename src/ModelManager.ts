import {
  AutoTokenizer,
  AutoModelForCausalLM,
  AutoModel,
  PreTrainedTokenizer,
  PreTrainedModel,
  pipeline
} from "@huggingface/transformers";

import { TTSEngine } from './TTSEngine';
import { BaseModule } from './BaseModule';
import { FileProgressTracker } from './FileProgressTracker';
import { WebGPUChecker } from './WebGPUChecker';
import { tryGarbageCollection, detectEnvironment } from './utils';
import {
  ModelType,
  BaseModelLoadOptions,
  GenerationModelLoadOptions,
  EmbeddingModelLoadOptions,
  AudioModelLoadOptions,
  GenerationModelInfo,
  EmbeddingModelInfo,
  AudioModelInfo,
  DeviceConfig
} from './types';

export interface ModelManagerOptions {
  progressTracker: any;
  webgpuChecker: WebGPUChecker;
}

/**
 * Manages model lifecycle for all types of models (generation, embedding, audio)
 */
export class ModelManager {
  protected progressTracker: any;
  protected webgpuChecker: WebGPUChecker;
  private generationModels: Map<string, GenerationModelInfo> = new Map();
  private embeddingModels: Map<string, EmbeddingModelInfo> = new Map();
  private audioModels: Map<string, AudioModelInfo> = new Map();
  private ttsEngine: TTSEngine;
  private modelIsLoading: boolean = false;
  private environment = detectEnvironment();

  constructor(options: ModelManagerOptions) {
    this.progressTracker = options.progressTracker;
    this.webgpuChecker = options.webgpuChecker;
    this.ttsEngine = new TTSEngine();
  }

  /**
   * Get optimal device configuration based on capabilities
   * @returns {Promise<DeviceConfig>} Device configuration
   */
  protected async getOptimalDeviceConfig(): Promise<DeviceConfig> {
    const capabilities = await this.webgpuChecker.check();
    return this.webgpuChecker.getOptimalConfig();
  }

  /**
   * Trigger garbage collection if available
   */
  protected triggerGC(): void {
    try {
      tryGarbageCollection();
    } catch (error) {
      // Ignore errors
    }
  }

  /**
   * Check if running in a Node.js environment
   * @returns {boolean} True if in Node.js environment
   */
  protected isNodeEnvironment(): boolean {
    return this.environment.isNode;
  }

  /**
   * Check if running in a browser environment
   * @returns {boolean} True if in browser environment
   */
  protected isBrowserEnvironment(): boolean {
    return this.environment.isBrowser;
  }

  /**
   * Load a model of any type
   */
  async loadModel(options: BaseModelLoadOptions): Promise<GenerationModelInfo | EmbeddingModelInfo | AudioModelInfo> {
    const { model, type = ModelType.Generation } = options;

    if (!model) {
      throw new Error('Model identifier is required');
    }

    if (this.modelIsLoading) {
      throw new Error('Another model is currently loading');
    }

    switch (type) {
      case ModelType.Generation:
        return this.loadGenerationModel(options as GenerationModelLoadOptions);
      case ModelType.Embedding:
        return this.loadEmbeddingModel(options as EmbeddingModelLoadOptions);
      case ModelType.Audio:
        const result = await this.loadAudioModel(options as AudioModelLoadOptions);
        return { loaded: result, model: this.ttsEngine };
      default:
        throw new Error(`Invalid model type: ${type}`);
    }
  }

  /**
   * Offload a model of any type
   */
  async offloadModel(options: BaseModelLoadOptions): Promise<boolean> {
    const { model, type = ModelType.Generation } = options;

    if (!model) {
      throw new Error('Model identifier is required');
    }

    switch (type) {
      case ModelType.Generation:
        return this.offloadGenerationModel(model);
      case ModelType.Embedding:
        return this.offloadEmbeddingModel(model);
      case ModelType.Audio:
        return this.offloadAudioModel(model);
      default:
        throw new Error(`Invalid model type: ${type}`);
    }
  }

  /**
   * Get loaded models of a specific type
   */
  getLoadedModels(type: ModelType = ModelType.Generation): string[] {
    switch (type) {
      case ModelType.Generation:
        return Array.from(this.generationModels.keys());
      case ModelType.Embedding:
        return Array.from(this.embeddingModels.keys());
      case ModelType.Audio:
        return Array.from(this.audioModels.entries())
          .filter(([_, info]) => info.loaded)
          .map(([model, _]) => model);
      default:
        throw new Error(`Invalid model type: ${type}`);
    }
  }

  /**
   * Get a specific model's info
   */
  getModelInfo(model: string, type: ModelType = ModelType.Generation): GenerationModelInfo | EmbeddingModelInfo | AudioModelInfo | undefined {
    switch (type) {
      case ModelType.Generation:
        return this.generationModels.get(model);
      case ModelType.Embedding:
        return this.embeddingModels.get(model);
      case ModelType.Audio:
        return this.audioModels.get(model);
      default:
        throw new Error(`Invalid model type: ${type}`);
    }
  }

  /**
   * Get TTS engine instance
   */
  getTTSEngine(): TTSEngine {
    return this.ttsEngine;
  }

  /**
   * Load a generation model
   */
  async loadGenerationModel(options: GenerationModelLoadOptions): Promise<GenerationModelInfo> {
    const { model, quantization } = options;

    if (!model) {
      throw new Error('Model identifier is required');
    }

    if (this.modelIsLoading) {
      throw new Error('Another model is currently loading');
    }

    if (this.generationModels.has(model)) {
      return this.generationModels.get(model)!;
    }

    this.modelIsLoading = true;

    try {
      const capabilities = await this.webgpuChecker.check();
      const config = await this.getOptimalDeviceConfig();

      // Create file tracker for detailed progress
      const fileTracker = new FileProgressTracker();
      const loadId = Date.now().toString();

      const trackedCallback = (progress: any) => {
        fileTracker.update(progress);
        const overall = fileTracker.getOverallProgress();

        this.progressTracker.update({
          status: 'loading',
          type: 'model',
          message: `Loading ${progress.component} for ${model}`,
          progress: overall.progress,
          percentComplete: overall.progress ? Math.round(overall.progress * 100) : undefined,
          loadId,
          modelId: model,
          files: fileTracker.getAllFiles(),
          overall,
          ...progress
        });
      };

      // Determine optimal model configuration
      const modelConfig: any = {};
      if (config.device) modelConfig.device = config.device;
      if (config.dtype) modelConfig.dtype = config.dtype;
      if (quantization) modelConfig.quantization = quantization;

      // Load tokenizer with progress tracking
      const tokenizerPromise = AutoTokenizer.from_pretrained(model, {
        progress_callback: (progress: any) => {
          const enhancedProgress = {
            ...progress,
            component: 'tokenizer'
          };
          trackedCallback(enhancedProgress);
        }
      });

      // Load model with progress tracking
      const modelPromise = AutoModelForCausalLM.from_pretrained(model, {
        ...modelConfig,
        progress_callback: (progress: any) => {
          const enhancedProgress = {
            ...progress,
            component: 'model'
          };
          trackedCallback(enhancedProgress);
        }
      });

      const [tokenizer, loadedModel] = await Promise.all([tokenizerPromise, modelPromise]);
      const registryEntry = { tokenizer, model: loadedModel };
      this.generationModels.set(model, registryEntry);

      this.progressTracker.update({
        status: 'ready',
        type: 'model',
        progress: 1,
        percentComplete: 100,
        message: `Model ${model} loaded successfully`,
        loadId,
        modelId: model,
        files: fileTracker.getAllFiles(),
        overall: fileTracker.getOverallProgress()
      });

      return registryEntry;
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      this.progressTracker.update({
        status: 'error',
        type: 'model',
        message: `Error loading model ${model}: ${errorMessage}`
      });
      throw new Error(`Failed to load model ${model}: ${errorMessage}`);
    } finally {
      this.modelIsLoading = false;
    }
  }

  /**
   * Load an embedding model
   */
  async loadEmbeddingModel(options: EmbeddingModelLoadOptions): Promise<EmbeddingModelInfo> {
    const { model, dimensions } = options;

    if (!model) {
      throw new Error('Model identifier is required');
    }

    if (this.embeddingModels.has(model)) {
      return this.embeddingModels.get(model)!;
    }

    this.progressTracker.update({
      status: 'loading',
      type: 'embedding_model',
      message: `Loading embedding model: ${model}`
    });

    try {
      const capabilities = await this.webgpuChecker.check();
      const config = await this.getOptimalDeviceConfig();

      let modelInfo: EmbeddingModelInfo;

      if (capabilities.isWebGPUSupported) {
        try {
          modelInfo = await this._loadWithTokenizerAndModel(model, config);
        } catch (directError) {
          console.warn('Direct model loading failed, falling back to pipeline:', directError);
          modelInfo = await this._loadWithPipeline(model, config);
        }
      } else {
        modelInfo = await this._loadWithPipeline(model, config);
      }

      if (dimensions && modelInfo.dimensions !== dimensions) {
        console.warn(`Requested ${dimensions} dimensions but model provides ${modelInfo.dimensions} dimensions`);
      }

      this.embeddingModels.set(model, modelInfo);

      this.progressTracker.update({
        status: 'ready',
        type: 'embedding_model',
        message: `Embedding model ${model} loaded successfully`
      });

      return modelInfo;
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      this.progressTracker.update({
        status: 'error',
        type: 'embedding_model',
        message: `Error loading embedding model ${model}: ${errorMessage}`
      });
      throw new Error(`Failed to load embedding model ${model}: ${errorMessage}`);
    }
  }

  /**
   * Load an audio model
   */
  async loadAudioModel(options: AudioModelLoadOptions): Promise<boolean> {
    const { model } = options;

    if (!model) {
      throw new Error('Model identifier is required');
    }

    if (this.modelIsLoading) {
      throw new Error('Another model is currently loading');
    }

    const info = this.audioModels.get(model);
    if (info?.loaded) {
      return true;
    }

    this.modelIsLoading = true;

    try {
      const capabilities = await this.webgpuChecker.check();
      const config = await this.getOptimalDeviceConfig();
      const isNode = this.isNodeEnvironment();
      const device = isNode ? "cpu" : !capabilities.isWebGPUSupported ? "wasm" : "webgpu";

      this.progressTracker.update({
        status: 'loading',
        type: 'tts_model',
        progress: 0,
        percentComplete: 0,
        message: `Loading TTS model ${model} (device="${device}", dtype="${"fp32"}")`
      });

      await this.ttsEngine.loadModel(model, {
        onProgress: (progress: any) => {
          this.progressTracker.update({
            status: 'loading',
            type: 'tts_model',
            message: `Loading TTS model: ${model}`,
            progress: progress.progress,
            percentComplete: progress.progress ? Math.round(progress.progress * 100) : undefined,
            ...progress
          });
        },
        device: device,
        dtype: "fp32",
      });

      this.audioModels.set(model, { loaded: true, model: this.ttsEngine });

      this.progressTracker.update({
        status: 'ready',
        type: 'tts_model',
        progress: 1,
        percentComplete: 100,
        message: `TTS model ${model} loaded successfully`
      });

      return true;
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      this.progressTracker.update({
        status: 'error',
        type: 'tts_model',
        message: `Error loading TTS model ${model}: ${errorMessage}`
      });
      throw new Error(`Failed to load TTS model ${model}: ${errorMessage}`);
    } finally {
      this.modelIsLoading = false;
    }
  }

  /**
   * Get a loaded generation model
   */
  getGenerationModel(model: string): GenerationModelInfo | undefined {
    return this.generationModels.get(model);
  }

  /**
   * Get a loaded embedding model
   */
  getEmbeddingModel(model: string): EmbeddingModelInfo | undefined {
    return this.embeddingModels.get(model);
  }

  /**
   * Check if an audio model is loaded
   */
  isAudioModelLoaded(model: string): boolean {
    const info = this.audioModels.get(model);
    return info?.loaded ?? false;
  }

  /**
   * Offload a generation model
   */
  async offloadGenerationModel(model: string): Promise<boolean> {
    if (!this.generationModels.has(model)) {
      return false;
    }

    try {
      this.generationModels.delete(model);
      this.triggerGC();
      return true;
    } catch (error) {
      console.error(`Error offloading generation model ${model}:`, error);
      return false;
    }
  }

  /**
   * Offload an embedding model
   */
  async offloadEmbeddingModel(model: string): Promise<boolean> {
    if (!this.embeddingModels.has(model)) {
      return false;
    }

    try {
      this.embeddingModels.delete(model);
      this.triggerGC();
      return true;
    } catch (error) {
      console.error(`Error offloading embedding model ${model}:`, error);
      return false;
    }
  }

  /**
   * Offload an audio model
   */
  async offloadAudioModel(model: string): Promise<boolean> {
    if (!this.audioModels.has(model)) {
      return false;
    }

    try {
      this.audioModels.delete(model);
      this.triggerGC();
      return true;
    } catch (error) {
      console.error(`Error offloading audio model ${model}:`, error);
      return false;
    }
  }

  private async _loadWithTokenizerAndModel(model: string, config: DeviceConfig): Promise<EmbeddingModelInfo> {
    const progressCallback = (component: string) => (progress: any) => {
      this.progressTracker.update({
        status: 'loading',
        type: 'embedding_model',
        message: `Loading ${component} for ${model}`,
        progress: progress.progress,
        percentComplete: progress.progress ? Math.round(progress.progress * 100) : undefined,
        component,
        ...progress
      });
    };

    const tokenizer = await AutoTokenizer.from_pretrained(model, {
      progress_callback: progressCallback('tokenizer')
    });

    const modelConfig: any = {};
    if (config.device) {
      modelConfig.device = config.device as "cpu" | "wasm" | "webgpu" | "auto" | "gpu" | "cuda" | "dml" | "webnn" | "webnn-npu" | "webnn-gpu" | "webnn-cpu";
    }
    if (config.dtype) {
      modelConfig.dtype = config.dtype;
    }

    const embeddingModel = await AutoModel.from_pretrained(model, {
      ...modelConfig,
      progress_callback: progressCallback('model')
    });

    const dimensions = this._getModelDimensions(embeddingModel);

    return {
      tokenizer,
      model: embeddingModel,
      dimensions
    };
  }

  private async _loadWithPipeline(model: string, config: DeviceConfig): Promise<EmbeddingModelInfo> {
    const modelConfig: any = {};
    if (config.device) {
      modelConfig.device = config.device as "cpu" | "wasm" | "webgpu" | "auto" | "gpu" | "cuda" | "dml" | "webnn" | "webnn-npu" | "webnn-gpu" | "webnn-cpu";
    }
    if (config.dtype) {
      modelConfig.dtype = config.dtype;
    }

    const embeddingPipeline = await pipeline('feature-extraction', model, {
      ...modelConfig,
      progress_callback: (progress: any) => {
        this.progressTracker.update({
          status: 'loading',
          type: 'embedding_model',
          message: `Loading pipeline for ${model}`,
          progress: progress.progress,
          percentComplete: progress.progress ? Math.round(progress.progress * 100) : undefined,
          ...progress
        });
      }
    });

    const dimensions = this._getModelDimensions(embeddingPipeline.model);

    return {
      tokenizer: embeddingPipeline.tokenizer,
      model: embeddingPipeline.model,
      pipeline: embeddingPipeline,
      dimensions
    };
  }

  private _getModelDimensions(model: any): number {
    // Try to get dimensions from model config or output size
    return model.config?.hidden_size ||
      model.config?.d_model ||
      model.config?.hidden_dim ||
      768; // Default fallback
  }
}
