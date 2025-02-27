/**
 * TinyLM - A lightweight wrapper over TransformerJS with OpenAI-compatible API
 * Incorporating WebGPU detection, efficient streaming, and progress tracking
 * Universal version that works in both browser and Node.js environments
 */

import {
  AutoTokenizer,
  AutoModelForCausalLM,
  TextStreamer,
  env,
  PreTrainedTokenizer,
  PreTrainedModel,
} from "@huggingface/transformers";
import { tryGarbageCollection } from "./gc";
import { CapabilityInfo, ChatMessage, CompletionChunk, CompletionOptions, CompletionResult, EnvInfo, FileInfo, GenerateOutput, GenerationFunctionParameters, InitOptions, ModelLoadOptions, ModelOutput, ModelRegistryEntry, OverallProgress, ProgressUpdate, TinyLMOptions, TokenizedInputs, WebGPUCapabilities } from "./types";
import { FileProgressTracker, ProgressTracker } from "./progress";
import { WebGPUChecker } from "./gpu";
import { GenerationController } from "./generation";

// WebGPU type declarations - only for browser environment
declare global {
  interface Navigator {
    gpu?: {
      requestAdapter(): Promise<GPUAdapter | null>;
    };
  }

  interface GPUAdapter {
    name: string;
    description?: string;
    features: Set<string>;
    limits: Record<string, any>;
  }

  // Don't explicitly declare gc to avoid conflicts with Node.js types
  // We'll use a safer approach to handle garbage collection
}


/**
 * Main TinyLM class that provides the OpenAI-compatible API
 */
export class TinyLM {
  private options: Required<TinyLMOptions>;
  private progressTracker: ProgressTracker;
  private webgpuChecker: WebGPUChecker;
  private controller: GenerationController;
  private activeModel: string | null = null;
  private tokenizer: PreTrainedTokenizer | null = null;
  private model: PreTrainedModel | null = null;
  private modelIsLoading: boolean = false;
  private initialized: boolean = false;
  private modelRegistry: Map<string, ModelRegistryEntry> = new Map();

  // API structure
  readonly chat: {
    completions: {
      create: (options: CompletionOptions) => Promise<CompletionResult | AsyncGenerator<CompletionChunk>>;
    };
  };

  readonly models: {
    load: (options: ModelLoadOptions) => Promise<ModelRegistryEntry>;
    offload: (options: { model: string }) => Promise<boolean>;
    interrupt: () => boolean;
    reset: () => void;
    check: () => Promise<CapabilityInfo>;
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
    this.controller = new GenerationController();

    // Create API structure similar to OpenAI
    this.chat = {
      completions: {
        create: this.createCompletion.bind(this)
      }
    };

    // Model management API
    this.models = {
      load: this.loadModel.bind(this),
      offload: this.offloadModel.bind(this),
      interrupt: this.interrupt.bind(this),
      reset: this.reset.bind(this),
      check: this.checkCapabilities.bind(this)
    };
  }

  /**
   * Check hardware capabilities
   * @returns {Promise<CapabilityInfo>} Hardware capabilities
   */
  async checkCapabilities(): Promise<CapabilityInfo> {
    const gpuCapabilities = await this.webgpuChecker.check();

    // Get CPU/GPU info from environment when available
    let envInfo: EnvInfo = {};
    try {
      envInfo = {
        backend: (env as any).backend || 'unknown',
        cpuInfo: (env as any).cpuInfo || 'unknown',
        gpuInfo: (env as any).gpuInfo || 'unknown'
      };
    } catch (error) {
      // Environment variables not available
    }

    return {
      ...gpuCapabilities,
      environment: envInfo,
      transformersVersion: env.version || 'unknown'
    };
  }

  /**
   * Initialize TinyLM with models
   * @param {InitOptions} options - Initialization options
   * @returns {Promise<TinyLM>} This instance
   */
  async init(options: InitOptions = {}): Promise<TinyLM> {
    const {
      models = [],
      lazyLoad = false
    } = options;

    if (!this.initialized) {
      // Check hardware capabilities
      const capabilities = await this.checkCapabilities();
      this.progressTracker.update({
        status: 'init',
        type: 'system',
        message: `Hardware check: WebGPU ${capabilities.isWebGPUSupported ? 'available' : 'not available'}`
      });

      // Load first model if specified and not using lazy loading
      if (models.length > 0 && !lazyLoad) {
        const modelToLoad = models[0];
        if (modelToLoad) {
          await this.loadModel({ model: modelToLoad });
        }
      } else if (models.length > 0) {
        // Just set the active model name without loading
        const modelToSet = models[0];
        if (modelToSet) {
          this.activeModel = modelToSet;
        }
      }

      this.initialized = true;
    }
    return this;
  }

  /**
   * Updated model loading method with detailed per-file progress tracking
   * @param {ModelLoadOptions} options - Load options
   * @returns {Promise<ModelRegistryEntry>} The loaded model
   */
  async loadModel(options: ModelLoadOptions): Promise<ModelRegistryEntry> {
    const { model, quantization } = options;

    if (!model) {
      throw new Error('Model identifier is required');
    }

    // Return if already loading
    if (this.modelIsLoading) {
      throw new Error('Another model is currently loading');
    }

    // Set as active and return if already loaded
    if (this.modelRegistry.has(model)) {
      this.activeModel = model;
      const registryEntry = this.modelRegistry.get(model)!;
      this.tokenizer = registryEntry.tokenizer;
      this.model = registryEntry.model;

      this.progressTracker.update({
        status: 'ready',
        type: 'model',
        progress: 1,
        percentComplete: 100,
        message: `Model ${model} is already loaded`
      });

      return registryEntry;
    }

    // Set loading state
    this.modelIsLoading = true;
    this.activeModel = model;

    try {
      // Check hardware capabilities
      const capabilities = await this.webgpuChecker.check();

      // Get optimal config (or use user-provided quantization)
      const config = this.webgpuChecker.getOptimalConfig();
      const modelConfig: Record<string, any> = {
        // Only specify device and dtype if we have definitive information
        ...(config.device ? { device: config.device } : {}),
        ...(config.dtype || quantization ? { dtype: quantization || config.dtype } : {})
      };

      // Initialize file progress tracker for this model load
      const fileTracker = new FileProgressTracker();

      // Create a unique tracker ID for this load operation
      const loadId = Date.now().toString();

      // Initial progress message
      this.progressTracker.update({
        status: 'loading',
        type: 'model',
        progress: 0,
        percentComplete: 0,
        message: `Loading model ${model}`,
        loadId,
        modelId: model,
        files: [] // Will be populated as files are discovered
      });

      // Function to send progress updates including file details
      const sendProgressUpdate = () => {
        const overall = fileTracker.getOverallProgress();
        const files = fileTracker.getAllFiles();

        // Create message based on overall progress
        let message = `Loading model ${model}`;
        if (overall.activeFileCount > 0) {
          message += ` (${overall.activeFileCount} files remaining)`;
          if (overall.formattedRemaining) {
            message += ` - ETA: ${overall.formattedRemaining}`;
          }
        }

        // Send detailed progress update including file list
        this.progressTracker.update({
          status: 'loading',
          type: 'model',
          progress: overall.progress,
          percentComplete: overall.percentComplete,
          message,
          loadId,
          modelId: model,
          files, // Complete list of file statuses
          overall // Overall progress stats
        });
      };

      // Interface for progress updates from Hugging Face
      interface HFProgress {
        file?: string;
        status?: string;
        progress?: number;
        total?: number;
        [key: string]: any;
      }

      // Custom progress tracking that handles file-level progress
      const trackedCallback = (progress: HFProgress) => {
        // Skip updates without a file property (not file-related)
        if (!progress.file) return;

        // Normalize the file ID (sometimes it's a URL, sometimes a filename)
        const fileId = progress.file.split('/').pop() || progress.file;

        // Process based on status
        if (progress.status === 'initiate') {
          // Register a new file
          fileTracker.registerFile(fileId, {
            name: progress.file,
            status: 'initiate',
            bytesTotal: progress.total || 0
          });
        }
        else if (progress.status === 'progress') {
          // Update progress for existing file
          fileTracker.updateFile(fileId, {
            status: 'progress',
            progress: progress.progress, // Using progress for compatibility
            total: progress.total, // Using total for compatibility
          });
        }
        else if (progress.status === 'done') {
          // Mark file as complete
          fileTracker.completeFile(fileId);
        }

        // Send progress update including all files
        sendProgressUpdate();
      };

      // Track loading progress
      let tokenizer: PreTrainedTokenizer, loadedModel: PreTrainedModel;

      // Load tokenizer with progress tracking
      const tokenizerPromise = AutoTokenizer.from_pretrained(model, {
        progress_callback: (progress: any) => {
          // Add component type to the progress update
          const enhancedProgress = {
            ...progress,
            component: 'tokenizer'
          };
          trackedCallback(enhancedProgress);
        }
      });

      // Load model with progress tracking
      const modelPromise = AutoModelForCausalLM.from_pretrained(model, {
        // Include device/dtype only if specifically determined
        ...modelConfig,
        progress_callback: (progress: any) => {
          // Add component type to the progress update
          const enhancedProgress = {
            ...progress,
            component: 'model'
          };
          trackedCallback(enhancedProgress);
        }
      });

      // Wait for both to complete
      [tokenizer, loadedModel] = await Promise.all([tokenizerPromise, modelPromise]);

      // Store models in registry
      this.modelRegistry.set(model, { tokenizer, model: loadedModel });

      // Set as current model
      this.tokenizer = tokenizer;
      this.model = loadedModel;

      // Final progress update - make sure to include the full file history
      const finalFiles = fileTracker.getAllFiles();
      const finalOverall = fileTracker.getOverallProgress();

      // Update progress and mark as loaded
      this.progressTracker.update({
        status: 'ready',
        type: 'model',
        progress: 1,
        percentComplete: 100,
        message: `Model ${model} loaded successfully`,
        loadId,
        modelId: model,
        files: finalFiles,
        overall: finalOverall
      });

      return { tokenizer, model: loadedModel };
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      this.progressTracker.update({
        status: 'error',
        type: 'model',
        message: `Error loading model ${model}: ${errorMessage}`
      });

      console.error(`Error loading model ${model}:`, error);
      throw new Error(`Failed to load model ${model}: ${errorMessage}`);
    } finally {
      this.modelIsLoading = false;
    }
  }

  /**
   * Offload a model from memory - works in both browser and Node.js
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
      type: 'model',
      message: `Offloading model ${model}`
    });

    try {
      // Remove from registry
      this.modelRegistry.delete(model);

      // Clear current model if it's the active one
      if (this.activeModel === model) {
        this.tokenizer = null;
        this.model = null;
        this.activeModel = null;
      }

      // Try to trigger garbage collection in any environment
      tryGarbageCollection();

      this.progressTracker.update({
        status: 'offloaded',
        type: 'model',
        message: `Model ${model} removed from memory`
      });

      return true;
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      this.progressTracker.update({
        status: 'error',
        type: 'model',
        message: `Error offloading model ${model}: ${errorMessage}`
      });

      return false;
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
   * Create a chat completion
   * @param {CompletionOptions} options - Completion options
   * @returns {Promise<CompletionResult>|AsyncGenerator<CompletionChunk>} Completion result or stream
   */
  async createCompletion(options: CompletionOptions): Promise<CompletionResult | AsyncGenerator<CompletionChunk>> {
    const {
      messages,
      stream = false,
      streamOptions = {},
      model = null,
      temperature = 0.7,
      max_tokens = 1024,
      do_sample = true,
      top_k = 40,
      top_p = 0.95,
      ...otherOptions
    } = options;

    // Ensure we're initialized
    if (!this.initialized) {
      throw new Error('TinyLM must be initialized before creating completions');
    }

    // Load model if specified and different from current
    if (model && model !== this.activeModel) {
      await this.loadModel({ model });
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
          temperature,
          max_new_tokens: max_tokens,
          do_sample,
          top_k,
          top_p,
          ...otherOptions
        }, streamOptions);
      } else {
        return this._generateResponse(inputs, {
          temperature,
          max_new_tokens: max_tokens,
          do_sample,
          top_k,
          top_p,
          ...otherOptions
        });
      }
    } finally {
      this.controller.isGenerating = false;
    }
  }

  /**
   * Generate a full response (non-streaming) without token counting
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

      // Extract just the assistant's response using a better approach
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

      // Format in OpenAI-compatible format without usage statistics
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
   * Stream a response in chunks with true real-time streaming
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

      // Import TextStreamer if not already imported
      const streamer = new TextStreamer(this.tokenizer!, {
        skip_prompt: true,
        skip_special_tokens: true
      });

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

      // Override the callback function to add to the queue
      const originalCallback = streamer.callback_function;
      streamer.callback_function = (text: string) => {
        if (text) {
          if (queueResolve) {
            queueResolve(text);
            queueResolve = null;
          } else {
            textQueue.push(text);
          }
        }

        if (originalCallback) {
          originalCallback(text);
        }
      };

      // Start the generation in the background
      const generatePromise = this.model!.generate(({
        ...inputs,
        past_key_values: this.controller.getPastKeyValues(),
        stopping_criteria: this.controller.getStoppingCriteria(),
        streamer,
        ...params
      } as unknown) as GenerationFunctionParameters).then((output: ModelOutput) => {
        // Save for potential continuation
        this.controller.setPastKeyValues(output.past_key_values);
        generationComplete = true;

        // Resolve any pending waiters with null (end of stream)
        if (queueResolve) {
          queueResolve(null);
        }
      }).catch(error => {
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
}

// Export types along with the class
export type {
  WebGPUCapabilities,
  FileInfo,
  OverallProgress,
  ProgressUpdate,
  ChatMessage,
  CompletionOptions,
  CompletionResult,
  CompletionChunk,
  CapabilityInfo,
  ModelRegistryEntry
};

// Default export
export default TinyLM;
