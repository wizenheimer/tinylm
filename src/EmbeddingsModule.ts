/**
 * Embeddings module for TinyLM
 * Provides OpenAI-compatible API for text embeddings
 */

import {
  AutoTokenizer,
  AutoModel,
  mean_pooling,
  layer_norm,
  pipeline
} from "@huggingface/transformers";

import { BaseModule } from './BaseModule';
import { EmbeddingCreateOptions, EmbeddingResult } from './types';

/**
 * Information about a loaded embedding model
 */
interface EmbeddingModelInfo {
  tokenizer: any;
  model: any;
  dimensions: number;
  // For pipeline-based approach
  pipeline?: any;
}

/**
 * Embeddings module for TinyLM
 */
export class EmbeddingsModule extends BaseModule {
  private embeddingModels: Map<string, EmbeddingModelInfo> = new Map();

  /**
   * Initialize the embeddings module
   * @param {Record<string, any>} options - Initialization options
   * @returns {Promise<void>} Initialization result
   */
  async init(options: Record<string, any> = {}): Promise<void> {
    await super.init(options);

    // Initialize any default models if specified
    const { defaultModel } = options;
    if (defaultModel) {
      try {
        await this._loadEmbeddingModel(defaultModel);
      } catch (error) {
        console.warn(`Failed to preload default embedding model: ${error}`);
      }
    }
  }

  /**
   * Create embeddings from text input
   * @param {EmbeddingCreateOptions} options - Embedding creation options
   * @returns {Promise<EmbeddingResult>} The embedding result
   */
  async create(options: EmbeddingCreateOptions): Promise<EmbeddingResult> {
    const {
      model,
      input,
      encoding_format = 'float',
      dimensions
    } = options;

    // Ensure model is a valid string
    if (!model || typeof model !== 'string') {
      throw new Error('Valid model identifier is required');
    }

    // Normalize input to array of strings
    const inputs = Array.isArray(input) ? input : [input];

    // Validate inputs are strings
    if (inputs.some(i => typeof i !== 'string')) {
      throw new Error('Input must be a string or an array of strings');
    }

    // Track embedding generation
    this.progressTracker.update({
      status: 'loading',
      type: 'embedding',
      message: `Generating embeddings with model: ${model}`
    });

    try {
      // Load embedding model if needed
      const modelInfo = await this._loadEmbeddingModel(model, dimensions);

      // Generate embeddings
      const embeddings = await this._generateEmbeddings(modelInfo, inputs);

      // Convert embeddings to the requested format
      const formattedEmbeddings = encoding_format === 'base64'
        ? this._convertToBase64(embeddings)
        : embeddings;

      // Calculate token usage (estimate)
      const tokenCounts = await this._countTokens(inputs, modelInfo.tokenizer);

      // Format response in OpenAI-compatible format
      const result: EmbeddingResult = {
        object: 'list',
        data: formattedEmbeddings.map((embedding, index) => ({
          object: 'embedding',
          embedding,
          index
        })),
        model,
        usage: {
          prompt_tokens: tokenCounts.prompt_tokens,
          total_tokens: tokenCounts.total_tokens
        }
      };

      this.progressTracker.update({
        status: 'complete',
        type: 'embedding',
        message: `Embedding generation complete`
      });

      return result;
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      this.progressTracker.update({
        status: 'error',
        type: 'embedding',
        message: `Embedding generation error: ${errorMessage}`
      });

      throw new Error(`Failed to generate embeddings: ${errorMessage}`);
    }
  }

  /**
   * Load an embedding model using the appropriate strategy
   * @private
   * @param {string} model - Model identifier
   * @param {number} dimensions - Optional dimensions to validate
   * @returns {Promise<EmbeddingModelInfo>} Model information
   */
  private async _loadEmbeddingModel(model: string, dimensions?: number): Promise<EmbeddingModelInfo> {
    // Check if model is already loaded
    if (this.embeddingModels.has(model)) {
      return this.embeddingModels.get(model)!;
    }

    this.progressTracker.update({
      status: 'loading',
      type: 'embedding_model',
      message: `Loading embedding model: ${model}`
    });

    try {
      // Check WebGPU capabilities and get optimal config
      const capabilities = await this.webgpuChecker.check();
      const config = await this.getOptimalDeviceConfig();

      let modelInfo: EmbeddingModelInfo;

      // Strategy based on WebGPU availability
      if (capabilities.isWebGPUSupported) {
        // WebGPU is available - try direct tokenizer+model approach first
        try {
          modelInfo = await this._loadWithTokenizerAndModel(model, config);
        } catch (directError) {
          // Direct approach failed, fall back to pipeline
          console.warn('Direct model loading failed, falling back to pipeline:', directError);
          modelInfo = await this._loadWithPipeline(model, config);
        }
      } else {
        // WebGPU not available - use pipeline approach for simplicity
        modelInfo = await this._loadWithPipeline(model, config);
      }

      // Validate dimensions if specified
      if (dimensions && modelInfo.dimensions !== dimensions) {
        console.warn(`Requested ${dimensions} dimensions but model provides ${modelInfo.dimensions} dimensions`);
      }

      // Store for reuse
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
   * Load model using direct tokenizer and model approach (preferred with WebGPU)
   * @private
   * @param {string} model - Model identifier
   * @param {any} config - Device configuration
   * @returns {Promise<EmbeddingModelInfo>} Model information
   */
  private async _loadWithTokenizerAndModel(model: string, config: any): Promise<EmbeddingModelInfo> {
    // Progress callback for loading
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

    // Load tokenizer
    const tokenizer = await AutoTokenizer.from_pretrained(model, {
      progress_callback: progressCallback('tokenizer')
    });

    // Load model with optimal configuration
    const embeddingModel = await AutoModel.from_pretrained(model, {
      ...(config.device ? { device: config.device } : {}),
      ...(config.dtype ? { dtype: config.dtype } : {}),
      progress_callback: progressCallback('model')
    });

    // Determine model dimensions
    const dimensions = this._getModelDimensions(embeddingModel);

    return {
      tokenizer,
      model: embeddingModel,
      dimensions
    };
  }

  /**
   * Load model using pipeline approach (fallback method)
   * @private
   * @param {string} model - Model identifier
   * @param {any} config - Device configuration
   * @returns {Promise<EmbeddingModelInfo>} Model information
   */
  private async _loadWithPipeline(model: string, config: any): Promise<EmbeddingModelInfo> {
    // Load using feature-extraction pipeline
    const embeddingPipeline = await pipeline('feature-extraction', model, {
      ...(config.device ? { device: config.device } : {}),
      ...(config.dtype ? { dtype: config.dtype } : {}),
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

    // Determine dimensions from the pipeline's model
    const dimensions = this._getModelDimensions(embeddingPipeline.model);

    return {
      tokenizer: embeddingPipeline.tokenizer,
      model: embeddingPipeline.model,
      pipeline: embeddingPipeline,
      dimensions
    };
  }

  /**
   * Generate embeddings for the input texts
   * @private
   * @param {EmbeddingModelInfo} modelInfo - Model information
   * @param {string[]} inputs - Input texts
   * @returns {Promise<number[][]>} Raw embeddings
   */
  private async _generateEmbeddings(modelInfo: EmbeddingModelInfo, inputs: string[]): Promise<number[][]> {
    // If we have a pipeline, use that for simplicity
    if (modelInfo.pipeline) {
      try {
        const output = await modelInfo.pipeline(inputs, {
          pooling: "mean",
          normalize: true
        });
        return output.tolist();
      } catch (error) {
        console.error('Pipeline embedding generation failed:', error);
        // Fall back to manual embedding generation
      }
    }

    // Manual embedding generation using tokenizer and model directly
    const { tokenizer, model } = modelInfo;

    // Process inputs in batches to avoid memory issues
    const batchSize = 8; // Adjust based on model size and available memory
    const results: number[][] = [];

    for (let i = 0; i < inputs.length; i += batchSize) {
      const batch = inputs.slice(i, i + batchSize);

      this.progressTracker.update({
        status: 'generating',
        type: 'embedding',
        message: `Processing batch ${Math.floor(i / batchSize) + 1}/${Math.ceil(inputs.length / batchSize)}`,
        progress: i / inputs.length
      });

      // Tokenize the batch
      const tokenized = tokenizer(batch, {
        padding: true,
        truncation: true
      });

      // Generate embeddings
      const output = await model(tokenized);

      // Get embeddings using mean pooling
      const embeddingsRaw = mean_pooling(
        output.last_hidden_state,
        tokenized.attention_mask
      );

      // Normalize embeddings (L2 norm)
      const normalizedEmbeddings = embeddingsRaw.normalize(2, -1).tolist();

      // Add to results
      results.push(...normalizedEmbeddings);
    }

    return results;
  }

  /**
   * Convert float embeddings to base64 encoding
   * @private
   * @param {number[][]} embeddings - Float embeddings
   * @returns {string[]} Base64 encoded embeddings
   */
  private _convertToBase64(embeddings: number[][]): string[] {
    return embeddings.map(embedding => {
      // Convert to Float32Array
      const float32Array = new Float32Array(embedding);

      // Convert to binary
      const buffer = new Uint8Array(float32Array.buffer);

      // Convert to base64
      return this._arrayBufferToBase64(buffer);
    });
  }

  /**
   * Helper to convert array buffer to base64
   * @private
   * @param {Uint8Array} buffer - Binary buffer
   * @returns {string} Base64 string
   */
  private _arrayBufferToBase64(buffer: Uint8Array): string {
    let binary = '';
    const bytes = new Uint8Array(buffer);
    const len = bytes.byteLength;

    for (let i = 0; i < len; i++) {
      binary += String.fromCharCode(bytes[i]!);
    }

    // Use btoa in browser, Buffer in Node.js
    if (typeof btoa === 'function') {
      return btoa(binary);
    } else if (typeof Buffer !== 'undefined') {
      return Buffer.from(binary, 'binary').toString('base64');
    }

    throw new Error('Base64 encoding not available in this environment');
  }

  /**
   * Get the output dimensions of a model
   * @private
   * @param {any} model - The model object
   * @returns {number} Model dimension
   */
  private _getModelDimensions(model: any): number {
    // Try to determine dimensions from model config or structure
    try {
      // Check model config
      if (model.config && model.config.hidden_size) {
        return model.config.hidden_size;
      }

      // Check config.d_model (common in some models)
      if (model.config && model.config.d_model) {
        return model.config.d_model;
      }

      // Check for special properties in sentence transformers
      if (model.config && model.config.sentence_embedding_dimension) {
        return model.config.sentence_embedding_dimension;
      }

      // Default dimensions for common embedding models
      return 768; // Common size for many transformer models
    } catch (e) {
      console.warn('Could not determine model dimensions, using default 768');
      return 768;
    }
  }

  /**
   * Count tokens in the input texts
   * @private
   * @param {string[]} inputs - Input texts
   * @param {any} tokenizer - Tokenizer
   * @returns {Promise<{prompt_tokens: number, total_tokens: number}>} Token counts
   */
  private async _countTokens(inputs: string[], tokenizer: any): Promise<{prompt_tokens: number, total_tokens: number}> {
    let prompt_tokens = 0;

    // Count tokens for each input
    for (const text of inputs) {
      try {
        const encoded = await tokenizer.encode(text);
        prompt_tokens += Array.isArray(encoded) ? encoded.length : 0;
      } catch (error) {
        // If tokenizer.encode fails, make a rough estimate
        prompt_tokens += Math.ceil(text.length / 4); // Rough approximation
      }
    }

    return {
      prompt_tokens,
      total_tokens: prompt_tokens
    };
  }

  /**
   * Offload an embedding model from memory
   * @param {string} model - Model identifier
   * @returns {Promise<boolean>} Success status
   */
  async offloadModel(model: string): Promise<boolean> {
    if (!this.embeddingModels.has(model)) {
      return false;
    }

    try {
      // Remove from registry
      this.embeddingModels.delete(model);

      // Try to trigger garbage collection
      this.triggerGC();

      this.progressTracker.update({
        status: 'offloaded',
        type: 'embedding_model',
        message: `Embedding model ${model} offloaded from memory`
      });

      return true;
    } catch (error) {
      console.error(`Error offloading embedding model ${model}:`, error);
      return false;
    }
  }
}
