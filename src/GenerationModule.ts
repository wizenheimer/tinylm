/**
 * Generation module for TinyLM
 * Encapsulates text generation functionality
 */

import {
  AutoTokenizer,
  AutoModelForCausalLM,
  PreTrainedTokenizer,
  PreTrainedModel
} from "@huggingface/transformers";

import { BaseModule } from './BaseModule';
import { GenerationController } from './GenerationController';
import { FileProgressTracker } from './FileProgressTracker';
import { createStreamer } from './TextStreamHandler';
import { ModelManager } from './ModelManager';
import {
  ModelRegistryEntry,
  TokenizedInputs,
  CompletionOptions,
  CompletionResult,
  CompletionChunk,
  ModelLoadOptions,
  ChatMessage,
  GenerationFunctionParameters,
  GenerateOutput,
  ModelOutput,
  ModelType,
  GenerationModelInfo,
  GenerationModelLoadOptions
} from './types';

/**
 * Generation module for handling text generation
 */
export class GenerationModule extends BaseModule {
  private controller: GenerationController;
  private activeModel: string | null = null;
  private tokenizer: PreTrainedTokenizer | null = null;
  private model: PreTrainedModel | null = null;
  private modelIsLoading: boolean = false;
  private modelRegistry: Map<string, ModelRegistryEntry> = new Map();
  private modelManager: ModelManager;

  /**
   * Create a new generation module
   * @param {any} tinyLM - Parent TinyLM instance
   */
  constructor(options: { modelManager: ModelManager }) {
    super(options);
    this.controller = new GenerationController();
    this.modelManager = options.modelManager;
  }

  /**
   * Initialize the generation module
   * @param {Record<string, any>} options - Initialization options
   * @returns {Promise<void>} Initialization result
   */
  async init(options: Record<string, any> = {}): Promise<void> {
    await super.init(options);

    const { models = [], lazyLoad = false } = options;

    // Check hardware capabilities
    const capabilities = await this.webgpuChecker.check();

    this.progressTracker.update({
      status: 'init',
      type: 'generation_module',
      message: `Hardware check: WebGPU ${capabilities.isWebGPUSupported ? 'available' : 'not available'}`
    });

    // Load first model if specified and not using lazy loading
    if (models.length > 0 && !lazyLoad) {
      const modelToLoad = models[0];
      if (modelToLoad) {
        await this.loadModel({
          model: modelToLoad,
          type: ModelType.Generation
        });
      }
    } else if (models.length > 0) {
      // Just set the active model name without loading
      const modelToSet = models[0];
      if (modelToSet) {
        this.activeModel = modelToSet;
      }
    }
  }

  /**
   * Load a model for text generation
   * @param {GenerationModelLoadOptions} options - Load options
   * @returns {Promise<GenerationModelInfo>} The loaded model
   */
  async loadModel(options: GenerationModelLoadOptions): Promise<GenerationModelInfo> {
    const { model } = options;

    if (!model) {
      throw new Error('Model identifier is required');
    }

    try {
      const registryEntry = await this.modelManager.loadGenerationModel(options);
      this.activeModel = model;
      this.tokenizer = registryEntry.tokenizer;
      this.model = registryEntry.model;
      return registryEntry;
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      throw new Error(`Failed to load model ${model}: ${errorMessage}`);
    }
  }

  /**
   * Get the currently active model
   * @returns {ModelRegistryEntry | undefined} The active model
   */
  getActiveModel(): ModelRegistryEntry | undefined {
    if (!this.activeModel) return undefined;
    return this.modelManager.getGenerationModel(this.activeModel);
  }

  /**
   * Offload a model from memory
   * @param {string} model - Model identifier
   * @returns {Promise<boolean>} Success status
   */
  async offloadModel(model: string): Promise<boolean> {
    const success = await this.modelManager.offloadGenerationModel(model);
    if (success && this.activeModel === model) {
      this.activeModel = null;
    }
    return success;
  }

  /**
   * Create a chat completion
   * @param {CompletionOptions} options - Completion options
   * @returns {Promise<CompletionResult>|AsyncGenerator<CompletionChunk>} Completion result or stream
   */
  async createCompletion(options: CompletionOptions): Promise<CompletionResult | AsyncGenerator<CompletionChunk>> {
    const { model = this.activeModel, messages, stream = false } = options;

    if (!model) {
      throw new Error('No model specified and no active model set');
    }

    // Load model if needed
    if (model !== this.activeModel) {
      await this.loadModel({
        model,
        type: ModelType.Generation
      });
    }

    // Ensure model is loaded
    if (!this.tokenizer || !this.model) {
      throw new Error('No model loaded. Call loadModel() first.');
    }

    // Format messages
    const inputs = this._formatMessages(messages);

    // Reset controller for new generation
    this.controller.reset();
    this.controller.isGenerating = true;

    try {
      if (stream) {
        return this._streamResponse(inputs, {
          temperature: 0.7,
          max_new_tokens: 1024,
          do_sample: true,
          top_k: 40,
          top_p: 0.95,
          ...options
        }, {});
      } else {
        return this._generateResponse(inputs, {
          temperature: 0.7,
          max_new_tokens: 1024,
          do_sample: true,
          top_k: 40,
          top_p: 0.95,
          ...options
        });
      }
    } finally {
      this.controller.isGenerating = false;
    }
  }

  /**
   * Format messages into a prompt
   * @private
   * @param {ChatMessage[]} messages - Array of message objects
   * @returns {TokenizedInputs} Tokenized input
   */
  private _formatMessages(messages: ChatMessage[]): TokenizedInputs {
    if (!this.tokenizer) {
      throw new Error('No tokenizer available. Please load a model first.');
    }

    // Use the tokenizer's chat template when available
    return this.tokenizer.apply_chat_template(messages, {
      add_generation_prompt: true,
      return_dict: true
    }) as TokenizedInputs;
  }

  /**
   * Generate a full response (non-streaming)
   * @private
   * @param {TokenizedInputs} inputs - Tokenized inputs
   * @param {Record<string, any>} params - Generation parameters
   * @returns {Promise<CompletionResult>} Completion in OpenAI format
   */
  private async _generateResponse(inputs: TokenizedInputs, params: Record<string, any>): Promise<CompletionResult> {
    const startTime = Date.now();

    this.progressTracker.update({
      status: 'generating',
      type: 'generation',
      message: `Generating response with model: ${this.activeModel}`
    });

    try {
      // Use cached values for continuation when available
      const past_key_values = this.controller.getPastKeyValues();

      // Generate with all the parameters
      const output = await this.model!.generate(({
        ...inputs,
        past_key_values: this.controller.getPastKeyValues(),
        stopping_criteria: this.controller.getStoppingCriteria(),
        return_dict_in_generate: true,
        ...params
      } as unknown) as GenerationFunctionParameters) as GenerateOutput;

      // Save for potential continuation
      this.controller.setPastKeyValues(output.past_key_values);

      // Decode the response
      const decoded = this.tokenizer!.batch_decode(output.sequences, {
        skip_special_tokens: true
      });

      const generatedText = decoded[0] || "";
      const timeTaken = Date.now() - startTime;

      // Extract just the assistant's response
      let responseText = '';
      try {
        // Try to match based on 'assistant' marker if present
        const assistantPattern = /assistant\s*([\s\S]+)$/i;
        const match = generatedText.match(assistantPattern);

        if (match && match[1]) {
          // Found the assistant's part
          responseText = match[1].trim();
        } else {
          // Fallback to prompt removal method
          const promptText = this.tokenizer!.batch_decode(
            Array.isArray(inputs.input_ids) && inputs.input_ids.length > 0 && inputs.input_ids[0]
              ? [inputs.input_ids[0]]
              : [[0]], // Use a default token ID if none available
            {
              skip_special_tokens: true
            }
          )[0] || '';

          if (generatedText && promptText && generatedText.startsWith(promptText)) {
            responseText = generatedText.slice(promptText.length).trim();
          } else {
            responseText = generatedText || '';
          }
        }
      } catch (error) {
        console.warn("Warning: Error extracting response:", error);
        responseText = generatedText;
      }

      // Log completion details
      this.progressTracker.update({
        status: 'complete',
        type: 'generation',
        message: `Generation complete (${timeTaken}ms)`
      });

      // Format in OpenAI-compatible format
      return {
        id: `chatcmpl-${Date.now()}`,
        object: 'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model: this.activeModel!,
        choices: [
          {
            index: 0,
            message: {
              role: 'assistant',
              content: responseText
            },
            finish_reason: 'stop'
          }
        ],
        _tinylm: {
          time_ms: timeTaken
        }
      };
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      this.progressTracker.update({
        status: 'error',
        type: 'generation',
        message: `Generation error: ${errorMessage}`
      });

      throw new Error(`Generation failed: ${errorMessage}`);
    }
  }

  /**
   * Stream a response in chunks
   * @private
   * @param {TokenizedInputs} inputs - Tokenized inputs
   * @param {Record<string, any>} params - Generation parameters
   * @param {Record<string, any>} streamOptions - Stream options
   * @returns {AsyncGenerator<CompletionChunk>} Stream of chunks
   */
  /**
   * Stream a response in chunks - cross-platform compatible implementation
   * Works in both browser and Node.js environments
   * @private
   * @param {TokenizedInputs} inputs - Tokenized inputs
   * @param {Record<string, any>} params - Generation parameters
   * @param {Record<string, any>} streamOptions - Stream options
   * @returns {AsyncGenerator<CompletionChunk>} Stream of chunks
   */
  private async *_streamResponse(
    inputs: TokenizedInputs,
    params: Record<string, any>,
    streamOptions: Record<string, any>
  ): AsyncGenerator<CompletionChunk> {
    const completionId = `chatcmpl-${Date.now()}`;

    try {
      const startTime = performance.now();

      if (!this.tokenizer) {
        throw new Error("Tokenizer is not initialized");
      }

      this.progressTracker.update({
        status: 'generating',
        type: 'generation',
        message: `Generating streaming response with model: ${this.activeModel}`
      });

      // Create a queue to store generated text chunks
      const textQueue: string[] = [];
      let queueResolve: ((value: string | null) => void) | null = null;
      let generationComplete = false;

      // Function to wait for the next chunk
      const waitForChunk = (): Promise<string | null> => {
        if (textQueue.length > 0) {
          return Promise.resolve(textQueue.shift()!);
        }

        if (generationComplete) {
          return Promise.resolve(null);
        }

        return new Promise<string | null>(resolve => {
          queueResolve = resolve;
        });
      };

      // Create a callback for text chunks
      const textCallback = (text: string) => {
        if (text) {
          if (queueResolve) {
            queueResolve(text);
            queueResolve = null;
          } else {
            textQueue.push(text);
          }
        }
      };

      // Create a streamer using our wrapper function
      const streamer = await createStreamer(this.tokenizer, {
        skip_prompt: true,
        skip_special_tokens: true,
        callback: textCallback
      });

      // Start the generation in the background
      const generatePromise = this.model!.generate({
        ...inputs,
        past_key_values: this.controller.getPastKeyValues(),
        stopping_criteria: this.controller.getStoppingCriteria(),
        streamer,
        ...params
      } as any).then((output: any) => {
        // Save for potential continuation
        if (output && output.past_key_values) {
          this.controller.setPastKeyValues(output.past_key_values);
        }
        generationComplete = true;

        // Resolve any pending waiters with null (end of stream)
        if (queueResolve) {
          queueResolve(null);
        }
      }).catch((error: any) => {
        console.error("Generation error:", error);
        generationComplete = true;
        if (queueResolve) {
          queueResolve(null);
        }
        throw error;
      });

      // Stream text chunks as they become available
      let chunk: string | null;
      while ((chunk = await waitForChunk()) !== null) {
        yield {
          id: completionId,
          object: 'chat.completion.chunk',
          created: Math.floor(Date.now() / 1000),
          model: this.activeModel!,
          choices: [
            {
              index: 0,
              delta: {
                content: chunk
              },
              finish_reason: null
            }
          ]
        };
      }

      // Ensure generation is complete
      await generatePromise;

      // Final chunk with finish reason
      yield {
        id: completionId,
        object: 'chat.completion.chunk',
        created: Math.floor(Date.now() / 1000),
        model: this.activeModel!,
        choices: [
          {
            index: 0,
            delta: {},
            finish_reason: 'stop'
          }
        ]
      };

      const totalTime = (performance.now() - startTime) / 1000;
      this.progressTracker.update({
        status: 'complete',
        type: 'generation',
        message: `Generation complete (${totalTime.toFixed(2)}s)`
      });

    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      this.progressTracker.update({
        status: 'error',
        type: 'generation',
        message: `Streaming error: ${errorMessage}`
      });

      // Yield error chunk
      yield {
        id: completionId,
        object: 'chat.completion.chunk',
        created: Math.floor(Date.now() / 1000),
        model: this.activeModel!,
        choices: [
          {
            index: 0,
            delta: {
              content: `\n[Error: ${errorMessage}]`
            },
            finish_reason: 'error'
          }
        ]
      };
    }
  }

  /**
   * Interrupt the current generation
   * @returns {boolean} True if generation was interrupted
   */
  interrupt(): boolean {
    const wasInterrupted = this.controller.interrupt();

    if (wasInterrupted) {
      this.progressTracker.update({
        status: 'interrupted',
        type: 'generation',
        message: 'Generation interrupted by user'
      });
    }

    return wasInterrupted;
  }

  /**
   * Reset the generation state
   */
  reset(): void {
    this.controller.reset();

    this.progressTracker.update({
      status: 'reset',
      type: 'generation',
      message: 'Generation state reset'
    });
  }

  /**
   * Get the model registry
   * @returns {Map<string, ModelRegistryEntry>} Model registry
   */
  getModelRegistry(): Map<string, ModelRegistryEntry> {
    return this.modelRegistry;
  }
}
