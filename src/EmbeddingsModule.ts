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
import { ModelManager } from './ModelManager';
import { EmbeddingCreateOptions, EmbeddingResult } from './types';
import { ModelType } from './types';

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
  private modelManager: ModelManager;
  private activeModel: string | null;

  /**
   * Create a new embeddings module
   * @param {Object} options - Module options
   * @param {ModelManager} options.modelManager - Model manager instance
   */
  constructor(options: { modelManager: ModelManager }) {
    super(options);
    this.modelManager = options.modelManager;
    this.activeModel = null;
  }

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
        await this.modelManager.loadEmbeddingModel(defaultModel);
      } catch (error) {
        console.warn(`Failed to preload default embedding model: ${error}`);
      }
    }
  }

  /**
   * Load an embedding model
   */
  async loadModel(model: string, dimensions?: number): Promise<EmbeddingModelInfo> {
    try {
      const modelInfo = await this.modelManager.loadEmbeddingModel({
        model,
        type: ModelType.Embedding,
        dimensions
      });
      this.activeModel = model;
      return modelInfo;
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      throw new Error(`Failed to load embedding model ${model}: ${errorMessage}`);
    }
  }

  /**
   * Create embeddings from input text
   */
  async create(options: EmbeddingCreateOptions): Promise<EmbeddingResult> {
    const { model = this.activeModel, input, encoding_format = 'float' } = options;

    if (!model) {
      throw new Error('No model specified and no active model set');
    }

    // Load model if needed
    const modelInfo = await this.loadModel(model);

    // Convert input to array if string
    const inputs = Array.isArray(input) ? input : [input];

    // Get embeddings
    const startTime = Date.now();
    const embeddings = await this._generateEmbeddings(modelInfo, inputs);
    const timeMs = Date.now() - startTime;

    // Calculate token usage (estimate)
    const tokenCounts = await this._countTokens(inputs, modelInfo.tokenizer);

    // Convert embeddings to the requested format if needed
    const formattedEmbeddings = encoding_format === 'base64'
      ? this._convertToBase64(embeddings)
      : embeddings;

    return {
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
      },
      _tinylm: {
        time_ms: timeMs,
        dimensions: modelInfo.dimensions
      }
    };
  }

  /**
   * Generate embeddings for the input texts
   * @private
   */
  private async _generateEmbeddings(modelInfo: any, inputs: string[]): Promise<number[][]> {
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
   * Convert embeddings to base64 format
   * @private
   */
  private _convertToBase64(embeddings: number[][]): string[] {
    return embeddings.map(embedding => {
      const buffer = new Float32Array(embedding).buffer;
      return Buffer.from(buffer).toString('base64');
    });
  }

  /**
   * Count tokens in input texts
   * @private
   */
  private async _countTokens(inputs: string[], tokenizer: any): Promise<{ prompt_tokens: number; total_tokens: number }> {
    const tokenCounts = await Promise.all(
      inputs.map(text => tokenizer.encode(text).length)
    );

    const total = tokenCounts.reduce((sum, count) => sum + count, 0);
    return {
      prompt_tokens: total,
      total_tokens: total
    };
  }

  /**
   * Offload a model from memory
   */
  async offloadModel(model: string): Promise<boolean> {
    return this.modelManager.offloadEmbeddingModel(model);
  }
}
