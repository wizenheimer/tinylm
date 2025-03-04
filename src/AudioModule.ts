import { BaseModule } from './BaseModule';
import { TTSEngine, AudioChunk } from './TTSEngine';
import { ModelManager } from './ModelManager';
import {
  SpeechCreateOptions,
  SpeechResult,
  SpeechStreamResult,
  ModelType
} from './types';
// import { splitTextIntoSentences, ensureSafeTokenLength } from './TextSplitter';

/**
 * Audio module for TinyLM
 */
export class AudioModule extends BaseModule {
  private modelManager: ModelManager;
  private activeModel: string | null = null;

  /**
   * Create a new audio module
   * @param {Object} options - Module options
   * @param {ModelManager} options.modelManager - Model manager instance
   */
  constructor(options: { modelManager: ModelManager }) {
    super(options);
    this.modelManager = options.modelManager;
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

    try {
      const success = await this.modelManager.loadAudioModel({
        model,
        type: ModelType.Audio
      });
      if (success) {
        this.activeModel = model;
      }
      return success;
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      throw new Error(`Failed to load TTS model ${model}: ${errorMessage}`);
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
      stream = false
    } = options;

    // Load model if specified and different from current
    if (model && model !== this.activeModel) {
      await this.loadModel({ model });
    }

    // Check if a model is loaded
    if (!this.activeModel || !this.modelManager.isAudioModelLoaded(this.activeModel)) {
      throw new Error('No TTS model loaded. Specify a model or call loadModel() first.');
    }

    this.progressTracker.update({
      status: 'generating',
      type: 'speech',
      message: `Generating speech for text with model: ${this.activeModel}`
    });

    try {
      const startTime = Date.now();
      const ttsEngine = this.modelManager.getTTSEngine();

      // Generate speech with or without streaming
      const result = await ttsEngine.generateSpeech(input, {
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
      if (Array.isArray(result)) {
        // Streaming mode - array of chunks
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
      } else {
        // Standard non-streaming mode - single audio buffer
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
   * Offload a TTS model from memory
   * @param {string} model - Model identifier
   * @returns {Promise<boolean>} Success status
   */
  async offloadModel(model: string): Promise<boolean> {
    const success = await this.modelManager.offloadAudioModel(model);
    if (success && this.activeModel === model) {
      this.activeModel = null;
    }
    return success;
  }

  /**
   * Get list of loaded TTS models
   * @returns {string[]} Array of model identifiers
   */
  getLoadedModels(): string[] {
    return this.modelManager.getLoadedModels(ModelType.Audio);
  }
}
