/**
 * Audio module for TinyLM
 * Provides OpenAI-compatible API for text-to-speech
 */

import { BaseModule } from './BaseModule';
import { AudioChunk, TTSEngine } from './TTSEngine';
import { SpeechCreateOptions, SpeechResult, SpeechStreamResult } from './types';
import { splitTextIntoSentences, ensureSafeTokenLength } from './TextSplitter';


/**
 * Audio module for TinyLM
 */
export class AudioModule extends BaseModule {
  private ttsEngine: TTSEngine;
  private activeModel: string | null = null;
  private modelRegistry: Map<string, boolean> = new Map();
  private modelIsLoading: boolean = false;

  /**
   * Create a new audio module
   * @param {any} tinyLM - Parent TinyLM instance
   */
  constructor(tinyLM: any) {
    super(tinyLM);
    this.ttsEngine = new TTSEngine();
  }

  /**
   * Initialize the audio module
   * @param {Record<string, any>} options - Initialization options
   * @returns {Promise<void>} Initialization result
   */
  async init(options: Record<string, any> = {}): Promise<void> {
    await super.init(options);

    const { ttsModels = [], lazyLoad = false } = options;

    // Check hardware capabilities
    const capabilities = await this.webgpuChecker.check();

    this.progressTracker.update({
      status: 'init',
      type: 'audio_module',
      message: `Hardware check: WebGPU ${capabilities.isWebGPUSupported ? 'available' : 'not available'}`
    });

    // Load first model if specified and not using lazy loading
    if (ttsModels.length > 0 && !lazyLoad) {
      const modelToLoad = ttsModels[0];
      if (modelToLoad) {
        await this.loadModel({ model: modelToLoad });
      }
    } else if (ttsModels.length > 0) {
      // Just set the active model name without loading
      const modelToSet = ttsModels[0];
      if (modelToSet) {
        this.activeModel = modelToSet;
      }
    }
  }

  /**
 * Load a TTS model
 * @param {Object} options - Load options
 * @returns {Promise<boolean>} Success status
 */
async loadModel(options: { model: string }): Promise<boolean> {
  const { model } = options;

  if (!model) {
    throw new Error('Model identifier is required');
  }

  // Return if already loading
  if (this.modelIsLoading) {
    throw new Error('Another model is currently loading');
  }

  // Set as active and return if already loaded
  if (this.modelRegistry.get(model) === true) {
    this.activeModel = model;

    this.progressTracker.update({
      status: 'ready',
      type: 'tts_model',
      progress: 1,
      percentComplete: 100,
      message: `Model ${model} is already loaded`
    });

    return true;
  }

  // Set loading state
  this.modelIsLoading = true;
  this.activeModel = model;

  try {
    // Check hardware capabilities
    const capabilities = await this.webgpuChecker.check();

    // Get optimal config
    const config = await this.getOptimalDeviceConfig();

    // Determine device explicitly based on environment and capabilities
    const device = this.isNodeEnvironment() ? "cpu" :
                  !capabilities.isWebGPUSupported ? "wasm" : "webgpu";

    // Initial progress message
    this.progressTracker.update({
      status: 'loading',
      type: 'tts_model',
      progress: 0,
      percentComplete: 0,
      message: `Loading TTS model ${model} (device="${device}", dtype="${config.dtype}")`
    });

    // Load the model with explicit device and dtype
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
      dtype: config.dtype
    });

    // Register the model as loaded
    this.modelRegistry.set(model, true);

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
   * Create speech from text
   * @param {SpeechCreateOptions} options - Speech creation options
   * @returns {Promise<SpeechResult|SpeechStreamResult>} Speech result
   */
  async createSpeech(options: SpeechCreateOptions): Promise<SpeechResult | SpeechStreamResult> {
    const {
      model,
      input,
      voice = 'af',
      response_format = 'mp3',
      speed = 1.0,
      stream = false // New streaming parameter
    } = options;

    // Load model if specified and different from current
    if (model && model !== this.activeModel) {
      await this.loadModel({ model });
    }

    // Check if a model is loaded
    if (!this.activeModel || !this.modelRegistry.get(this.activeModel)) {
      throw new Error('No TTS model loaded. Specify a model or call loadModel() first.');
    }

    this.progressTracker.update({
      status: 'generating',
      type: 'speech',
      message: `Generating speech for text with model: ${this.activeModel}`
    });

    try {
      const startTime = Date.now();

      // Generate speech with or without streaming
      const result = await this.ttsEngine.generateSpeech(input, {
        voice,
        speed,
        stream
      });

      const timeTaken = Date.now() - startTime;

      this.progressTracker.update({
        status: 'complete',
        type: 'speech',
        message: `Speech generation complete (${timeTaken}ms)`
      });

      // Handle the result based on streaming mode
      if (!stream) {
        // Standard non-streaming mode - return single audio buffer
        return {
          id: `speech-${Date.now()}`,
          object: 'audio.speech',
          created: Math.floor(Date.now() / 1000),
          model: this.activeModel!,
          audio: result as ArrayBuffer,
          content_type: response_format === 'mp3' ? 'audio/mpeg' : 'audio/wav',
          _tinylm: {
            time_ms: timeTaken
          }
        };
      } else {
        // Streaming mode - return array of chunks
        const audioChunks = result as AudioChunk[];

        return {
          id: `speech-stream-${Date.now()}`,
          object: 'audio.speech.stream',
          created: Math.floor(Date.now() / 1000),
          model: this.activeModel!,
          chunks: audioChunks.map(chunk => ({
            text: chunk.text,
            audio: chunk.audio,
            content_type: response_format === 'mp3' ? 'audio/mpeg' : 'audio/wav'
          })),
          _tinylm: {
            time_ms: timeTaken
          }
        };
      }
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      this.progressTracker.update({
        status: 'error',
        type: 'speech',
        message: `Speech generation error: ${errorMessage}`
      });

      throw new Error(`Failed to generate speech: ${errorMessage}`);
    }
  }

  /**
   * Create a combined speech from multiple sentences
   * @param {string} text - Full text to process
   * @param {any} options - Generation options
   * @returns {Promise<ArrayBuffer>} Combined audio
   */
  async createCombinedSpeech(text: string, options: {
    voice?: string;
    speed?: number;
  } = {}): Promise<ArrayBuffer> {
    const { voice = 'af', speed = 1.0 } = options;

    // Check if a model is loaded
    if (!this.activeModel || !this.modelRegistry.get(this.activeModel)) {
      throw new Error('No TTS model loaded.');
    }

    // Split text into sentences
    const sentences = splitTextIntoSentences(text);

    // Generate combined speech
    return this.ttsEngine.generateCombinedSpeech(sentences, voice, speed);
  }


  /**
   * Offload a TTS model from memory
   * @param {Object} options - Offload options
   * @returns {Promise<boolean>} Success status
   */
  async offloadModel(options: { model: string }): Promise<boolean> {
    const { model } = options;

    if (!model) {
      throw new Error('Model identifier is required');
    }

    if (!this.modelRegistry.has(model)) {
      return false;
    }

    this.progressTracker.update({
      status: 'offloading',
      type: 'tts_model',
      message: `Offloading TTS model ${model}`
    });

    try {
      // Remove from registry
      this.modelRegistry.delete(model);

      // Clear current model if it's the active one
      if (this.activeModel === model) {
        this.activeModel = null;
      }

      // Try to trigger garbage collection
      this.triggerGC();

      this.progressTracker.update({
        status: 'offloaded',
        type: 'tts_model',
        message: `TTS model ${model} removed from memory`
      });

      return true;
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      this.progressTracker.update({
        status: 'error',
        type: 'tts_model',
        message: `Error offloading TTS model ${model}: ${errorMessage}`
      });

      return false;
    }
  }

  /**
   * Get active model identifier
   * @returns {string|null} Active model identifier
   */
  getActiveModel(): string | null {
    return this.activeModel;
  }

  /**
   * Get list of loaded TTS models
   * @returns {string[]} Array of model identifiers
   */
  getLoadedModels(): string[] {
    return Array.from(this.modelRegistry.entries())
      .filter(([_, loaded]) => loaded)
      .map(([model, _]) => model);
  }
}
