/**
 * TinyLM - A lightweight wrapper over TransformerJS with OpenAI-compatible API
 * Incorporating WebGPU detection, efficient streaming, and progress tracking
 */

import {
  AutoTokenizer,
  AutoModelForCausalLM,
  TextStreamer,
  env,
  PreTrainedTokenizer,
  PreTrainedModel,
} from "@huggingface/transformers";

// WebGPU type declarations
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

  var gc: undefined | (() => void);
}

// Custom types for transformers.js
export interface ModelOutput {
  sequences?: number[][];
  past_key_values?: any;
  [key: string]: any;
}

export interface GenerateOutput extends ModelOutput {
  sequences: number[][];
  past_key_values?: any;
}

export interface StoppingCriteriaList {
  criteria: any[];
  push: (criterion: any) => void;
  extend: (criteria: any[]) => void;
  [Symbol.iterator](): IterableIterator<any>;
  shouldStop: (input_ids: number[][], scores: number[][], options?: any) => boolean;
}

export interface GenerationFunctionParameters {
  input_ids?: number[][];
  attention_mask?: number[][];
  max_new_tokens?: number;
  temperature?: number;
  top_k?: number;
  top_p?: number;
  repetition_penalty?: number;
  stopping_criteria?: any;
  streamer?: any;
  return_dict_in_generate?: boolean;
  past_key_values?: any;
  [key: string]: any; // Allow additional properties
}

// Update InterruptableStoppingCriteria to implement StoppingCriteriaList
class InterruptableStoppingCriteria implements StoppingCriteriaList {
  criteria: any[] = [];
  private interrupted: boolean = false;

  push(criterion: any): void {
    this.criteria.push(criterion);
  }

  extend(criteria: any[]): void {
    this.criteria.push(...criteria);
  }

  [Symbol.iterator](): IterableIterator<any> {
    return this.criteria[Symbol.iterator]();
  }

  shouldStop(input_ids: number[][], scores: number[][], options?: any): boolean {
    return this.interrupted;
  }

  interrupt(): void {
    this.interrupted = true;
  }

  reset(): void {
    this.interrupted = false;
  }

  /**
   * Create a proxy that makes this object callable for transformers.js
   */
  asCallable(): any {
    const self = this;
    return new Proxy(this, {
      apply(_target: any, _thisArg: any, args: [number[][], number[][], ...any[]]): any[] {
        return [self.shouldStop(args[0], args[1], args[2])];
      },
      get(target: any, prop: string | symbol): any {
        if (prop === Symbol.iterator) {
          return target[Symbol.iterator].bind(target);
        }
        return target[prop];
      }
    });
  }
}

/**
 * WebGPU feature detection
 * Checks if WebGPU is supported and which features are available
 */
export interface WebGPUCapabilities {
  isWebGPUSupported: boolean;
  fp16Supported: boolean;
  isNode?: boolean;
  reason?: string;
  adapterInfo?: {
    name: string;
    description: string;
    features: string[];
    limits: Record<string, any>;
  };
}

export interface DeviceConfig {
  device?: string;
  dtype?: string;
}

class WebGPUChecker {
  private isWebGPUSupported: boolean = false;
  private fp16Supported: boolean = false;
  private adapter: GPUAdapter | null = null;

  /**
   * Check if WebGPU is supported in the current environment
   * @returns {Promise<WebGPUCapabilities>} WebGPU capabilities
   */
  async check(): Promise<WebGPUCapabilities> {
    try {
      // Skip WebGPU check in Node.js environment
      const isNode = typeof process !== 'undefined' &&
        process.versions != null &&
        process.versions.node != null;

      if (isNode) {
        return {
          isWebGPUSupported: false,
          fp16Supported: false,
          isNode: true,
          reason: "Running in Node.js environment"
        };
      }

      // Check for browser WebGPU support
      if (typeof navigator === 'undefined' || !navigator.gpu) {
        return {
          isWebGPUSupported: false,
          fp16Supported: false,
          reason: "WebGPU is not available in this environment"
        };
      }

      const adapter = await navigator.gpu.requestAdapter();
      if (!adapter) {
        return {
          isWebGPUSupported: false,
          fp16Supported: false,
          reason: "WebGPU is not supported (no adapter found)"
        };
      }

      this.adapter = adapter;
      this.isWebGPUSupported = true;
      this.fp16Supported = adapter.features.has("shader-f16");

      return {
        isWebGPUSupported: this.isWebGPUSupported,
        fp16Supported: this.fp16Supported,
        adapterInfo: {
          name: adapter.name,
          description: adapter.description || "No description available",
          features: Array.from(adapter.features),
          limits: { ...adapter.limits }
        }
      };
    } catch (error) {
      const errorMessage = error instanceof Error ? error.toString() : String(error);
      return {
        isWebGPUSupported: false,
        fp16Supported: false,
        reason: errorMessage
      };
    }
  }

  /**
   * Get the optimal device and configuration based on capabilities
   * @returns {DeviceConfig} Device configuration
   */
  getOptimalConfig(): DeviceConfig {
    // In Node.js or without WebGPU, let Transformers.js decide
    if (!this.isWebGPUSupported) {
      return {
        device: undefined, // Let Transformers.js decide the best device
        dtype: undefined   // Let Transformers.js decide the best dtype
      };
    }

    // With WebGPU available, use it with appropriate quantization
    return {
      device: "webgpu",
      dtype: this.fp16Supported ? "q4f16" : "q4"
    };
  }
}

/**
 * Interface for file information used by FileProgressTracker
 */
export interface FileInfo {
  id: string;
  name: string;
  status: string;
  progress: number;
  percentComplete: number;
  bytesLoaded: number;
  bytesTotal: number;
  startTime: number;
  lastUpdateTime: number;
  speed: number; // bytes per second
  timeRemaining: number | null; // in seconds
  [key: string]: any;
}

/**
 * Interface for overall progress information
 */
export interface OverallProgress {
  progress: number;
  percentComplete: number;
  bytesLoaded: number;
  bytesTotal: number;
  activeFileCount: number;
  totalFileCount: number;
  speed: number;
  timeRemaining: number;
  formattedLoaded: string;
  formattedTotal: string;
  formattedSpeed: string;
  formattedRemaining: string;
  isComplete: boolean;
  hasError: boolean;
}

/**
 * FileProgressTracker - Manages per-file progress tracking
 * Tracks the status and progress of individual files during model loading
 */
class FileProgressTracker {
  private files: Map<string, FileInfo> = new Map();
  private totalBytes: number = 0;
  private loadedBytes: number = 0;

  /**
   * Register a new file for tracking
   * @param {string} fileId - Unique file identifier
   * @param {Partial<FileInfo>} info - File information
   * @returns {FileInfo} File information object
   */
  registerFile(fileId: string, info: Partial<FileInfo> = {}): FileInfo {
    if (!this.files.has(fileId)) {
      const fileInfo: FileInfo = {
        id: fileId,
        name: info.name || fileId,
        status: 'pending',
        progress: 0,
        percentComplete: 0,
        bytesLoaded: 0,
        bytesTotal: info.bytesTotal || 0,
        startTime: Date.now(),
        lastUpdateTime: Date.now(),
        speed: 0, // bytes per second
        timeRemaining: null, // in seconds
        ...info
      };

      this.files.set(fileId, fileInfo);

      if (fileInfo.bytesTotal > 0) {
        this.totalBytes += fileInfo.bytesTotal;
      }

      return fileInfo;
    }

    return this.files.get(fileId)!;
  }

  /**
   * Update progress for a specific file
   * @param {string} fileId - File identifier
   * @param {Partial<FileInfo>} update - Progress update
   * @returns {FileInfo} Updated file info
   */
  updateFile(fileId: string, update: Partial<FileInfo>): FileInfo {
    if (!this.files.has(fileId)) {
      return this.registerFile(fileId, update);
    }

    const fileInfo = this.files.get(fileId)!;
    const now = Date.now();
    const timeDelta = (now - fileInfo.lastUpdateTime) / 1000; // seconds

    // Calculate bytes delta for speed estimation
    let bytesDelta = 0;
    if (update.bytesLoaded !== undefined && update.bytesTotal !== undefined) {
      // If we have new progress/total info
      const newBytesLoaded = update.bytesLoaded;
      bytesDelta = newBytesLoaded - fileInfo.bytesLoaded;

      // Update total bytes if it changed
      if (update.bytesTotal !== fileInfo.bytesTotal && update.bytesTotal > 0) {
        this.totalBytes = this.totalBytes - fileInfo.bytesTotal + update.bytesTotal;
        fileInfo.bytesTotal = update.bytesTotal;
      }

      // Update loaded bytes
      this.loadedBytes = this.loadedBytes - fileInfo.bytesLoaded + newBytesLoaded;
      fileInfo.bytesLoaded = newBytesLoaded;

      // Calculate percentage
      if (fileInfo.bytesTotal > 0) {
        fileInfo.progress = fileInfo.bytesLoaded / fileInfo.bytesTotal;
        fileInfo.percentComplete = Math.round(fileInfo.progress * 100);
      }
    }

    // Calculate speed (bytes per second) with some smoothing
    if (timeDelta > 0 && bytesDelta > 0) {
      const instantSpeed = bytesDelta / timeDelta;
      // Smooth speed calculation (70% previous, 30% new)
      fileInfo.speed = fileInfo.speed === 0
        ? instantSpeed
        : (fileInfo.speed * 0.7) + (instantSpeed * 0.3);

      // Calculate time remaining
      if (fileInfo.speed > 0 && fileInfo.bytesTotal > fileInfo.bytesLoaded) {
        const bytesRemaining = fileInfo.bytesTotal - fileInfo.bytesLoaded;
        fileInfo.timeRemaining = bytesRemaining / fileInfo.speed;
      }
    }

    // Update status
    if (update.status) {
      fileInfo.status = update.status;

      if (update.status === 'done' || update.status === 'complete') {
        fileInfo.progress = 1;
        fileInfo.percentComplete = 100;
        fileInfo.timeRemaining = 0;

        // Ensure consistency with bytes
        if (fileInfo.bytesTotal > 0 && fileInfo.bytesLoaded !== fileInfo.bytesTotal) {
          this.loadedBytes = this.loadedBytes - fileInfo.bytesLoaded + fileInfo.bytesTotal;
          fileInfo.bytesLoaded = fileInfo.bytesTotal;
        }
      }
    }

    // Update any other properties
    Object.assign(fileInfo, update);
    fileInfo.lastUpdateTime = now;

    return fileInfo;
  }

  /**
   * Mark a file as complete
   * @param {string} fileId - File identifier
   * @returns {FileInfo | null} Updated file info or null if not found
   */
  completeFile(fileId: string): FileInfo | null {
    if (!this.files.has(fileId)) {
      return null;
    }

    return this.updateFile(fileId, {
      status: 'done',
      progress: 1,
      percentComplete: 100,
      timeRemaining: 0
    });
  }

  /**
   * Get information for a specific file
   * @param {string} fileId - File identifier
   * @returns {FileInfo|null} File info or null if not found
   */
  getFile(fileId: string): FileInfo | null {
    return this.files.has(fileId) ? this.files.get(fileId)! : null;
  }

  /**
   * Get information about all tracked files
   * @returns {FileInfo[]} Array of file information objects
   */
  getAllFiles(): FileInfo[] {
    return Array.from(this.files.values());
  }

  /**
   * Get active (incomplete) files
   * @returns {FileInfo[]} Array of active file information objects
   */
  getActiveFiles(): FileInfo[] {
    return this.getAllFiles().filter(file =>
      file.status !== 'done' && file.status !== 'error');
  }

  /**
   * Get overall progress across all files
   * @returns {OverallProgress} Overall progress information
   */
  getOverallProgress(): OverallProgress {
    const files = this.getAllFiles();
    const activeFiles = this.getActiveFiles();

    // Calculate weighted overall progress
    let overallProgress = 0;
    if (this.totalBytes > 0) {
      overallProgress = this.loadedBytes / this.totalBytes;
    } else if (files.length > 0) {
      // Fallback to simple average if we don't have byte information
      const progressSum = files.reduce((sum, file) => sum + file.progress, 0);
      overallProgress = progressSum / files.length;
    }

    const percentComplete = Math.round(overallProgress * 100);

    // Calculate overall speed and time remaining
    let overallSpeed = 0;
    let maxTimeRemaining = 0;

    activeFiles.forEach(file => {
      overallSpeed += file.speed || 0;
      if (file.timeRemaining !== null && file.timeRemaining > maxTimeRemaining) {
        maxTimeRemaining = file.timeRemaining;
      }
    });

    // Format for human-readable size and time
    const formatBytes = (bytes: number): string => {
      if (bytes === 0) return '0 B';
      const sizes = ['B', 'KB', 'MB', 'GB'];
      const i = Math.floor(Math.log(bytes) / Math.log(1024));
      return `${(bytes / Math.pow(1024, i)).toFixed(2)} ${sizes[i]}`;
    };

    const formatTime = (seconds: number | null): string => {
      if (!seconds || seconds === 0) return '';
      if (seconds < 60) return `${Math.ceil(seconds)}s`;
      if (seconds < 3600) {
        const minutes = Math.floor(seconds / 60);
        const secs = Math.ceil(seconds % 60);
        return `${minutes}m ${secs}s`;
      }
      const hours = Math.floor(seconds / 3600);
      const minutes = Math.floor((seconds % 3600) / 60);
      return `${hours}h ${minutes}m`;
    };

    return {
      progress: overallProgress,
      percentComplete,
      bytesLoaded: this.loadedBytes,
      bytesTotal: this.totalBytes,
      activeFileCount: activeFiles.length,
      totalFileCount: files.length,
      speed: overallSpeed, // bytes per second
      timeRemaining: maxTimeRemaining, // most conservative estimate (longest file)

      // Human readable formats
      formattedLoaded: formatBytes(this.loadedBytes),
      formattedTotal: formatBytes(this.totalBytes),
      formattedSpeed: formatBytes(overallSpeed) + '/s',
      formattedRemaining: formatTime(maxTimeRemaining),

      // Status flags
      isComplete: activeFiles.length === 0 && files.length > 0,
      hasError: files.some(file => file.status === 'error')
    };
  }

  /**
   * Reset tracker state
   */
  reset(): void {
    this.files.clear();
    this.totalBytes = 0;
    this.loadedBytes = 0;
  }
}

/**
 * Interface for progress update
 */
export interface ProgressUpdate {
  status?: string;
  progress?: number;
  percentComplete?: number;
  message?: string;
  type?: string;
  [key: string]: any;
}

/**
 * Interface for progress tracker options
 */
interface ProgressTrackerOptions {
  throttleTime?: number;
  significantChangeThreshold?: number;
  [key: string]: any;
}

/**
 * Progress tracker with throttling and percentage support
 */
class ProgressTracker {
  private callback: (progress: ProgressUpdate) => void;
  private options: Required<ProgressTrackerOptions>;
  private lastUpdateTime: number = 0;
  private lastProgress: ProgressUpdate = {
    status: undefined,
    progress: undefined,
    percentComplete: undefined,
    message: undefined,
    type: undefined
  };

  /**
   * Create a new progress tracker
   * @param {Function} callback - Progress callback function
   * @param {ProgressTrackerOptions} options - Throttling options
   */
  constructor(
    callback?: (progress: ProgressUpdate) => void,
    options: ProgressTrackerOptions = {}
  ) {
    this.callback = callback || (() => {});
    this.options = {
      throttleTime: options.throttleTime || 100, // ms between updates
      significantChangeThreshold: options.significantChangeThreshold || 0.01, // 1% change
      ...options
    } as Required<ProgressTrackerOptions>;
  }

  /**
   * Update progress with throttling
   * @param {ProgressUpdate} progress - Progress object
   */
  update(progress: ProgressUpdate): void {
    try {
      // Sanitize progress value
      const sanitizedProgress: ProgressUpdate = { ...progress };

      // Ensure progress is between 0-1 if provided
      if (typeof progress.progress === 'number') {
        sanitizedProgress.progress = Math.max(0, Math.min(1, progress.progress));
      }

      // Ensure percentComplete is between 0-100 if provided
      if (typeof progress.percentComplete === 'number') {
        sanitizedProgress.percentComplete = Math.max(0, Math.min(100,
          Math.round(progress.percentComplete)));
      } else if (typeof progress.progress === 'number') {
        // Convert progress to percentage if not provided
        sanitizedProgress.percentComplete = Math.round((progress.progress || 0) * 100);
      }

      const now = Date.now();
      const timeSinceLastUpdate = now - this.lastUpdateTime;

      // Determine if update is significant
      const isSignificant = this._isSignificantChange(sanitizedProgress);
      const isStatusChange = sanitizedProgress.status !== this.lastProgress.status;
      const isFirstUpdate = this.lastUpdateTime === 0;
      const isLastUpdate = [
        'ready', 'done', 'error', 'complete', 'interrupted'
      ].includes(sanitizedProgress.status || '');

      if (
        isFirstUpdate ||
        isLastUpdate ||
        isStatusChange ||
        (isSignificant && timeSinceLastUpdate >= this.options.throttleTime)
      ) {
        this.callback(sanitizedProgress);
        this.lastUpdateTime = now;
        this.lastProgress = { ...sanitizedProgress };
      }
    } catch (error) {
      console.error("Error in progress tracker:", error);
    }
  }

  /**
   * Check if a progress update represents a significant change
   * @private
   * @param {ProgressUpdate} progress - Progress object
   * @returns {boolean} True if change is significant
   */
  private _isSignificantChange(progress: ProgressUpdate): boolean {
    if (progress.status !== this.lastProgress.status) return true;
    if (progress.type !== this.lastProgress.type) return true;
    if (progress.message !== this.lastProgress.message) return true;

    // Check either percentage or progress for significance
    if (
      typeof progress.percentComplete === 'number' &&
      typeof this.lastProgress.percentComplete === 'number'
    ) {
      const percentDiff = Math.abs(progress.percentComplete - this.lastProgress.percentComplete);
      return percentDiff >= 1; // 1% change is significant
    } else if (
      typeof progress.progress === 'number' &&
      typeof this.lastProgress.progress === 'number'
    ) {
      const progressDiff = Math.abs(progress.progress - this.lastProgress.progress);
      return progressDiff >= this.options.significantChangeThreshold;
    }

    return false;
  }
}

/**
 * Interruptable generation with the ability to stop generation midway
 */
class GenerationController {
  stoppingCriteria: InterruptableStoppingCriteria;
  isGenerating: boolean = false;
  private pastKeyValuesCache: any = null;

  constructor() {
    this.stoppingCriteria = new InterruptableStoppingCriteria();
  }

  /**
   * Interrupt the current generation
   * @returns {boolean} Whether generation was successfully interrupted
   */
  interrupt(): boolean {
    if (this.isGenerating) {
      this.stoppingCriteria.interrupt();
      this.isGenerating = false;
      return true;
    }
    return false;
  }

  /**
   * Reset the controller state
   */
  reset(): void {
    this.stoppingCriteria.reset();
    this.pastKeyValuesCache = null;
    this.isGenerating = false;
  }

  /**
   * Get the stopping criteria for generation
   * @returns {any} Stopping criteria object
   */
  getStoppingCriteria(): any {
    return this.stoppingCriteria.asCallable();
  }

  /**
   * Get cached key values from previous generation
   * @returns {any|null} Past key values or null if not available
   */
  getPastKeyValues(): any {
    return this.pastKeyValuesCache;
  }

  /**
   * Set cached key values from generation
   * @param {any} pastKeyValues - Key values to cache
   */
  setPastKeyValues(pastKeyValues: any): void {
    this.pastKeyValuesCache = pastKeyValues;
  }
}

/**
 * Interface for TinyLM constructor options
 */
export interface TinyLMOptions {
  progressCallback?: (progress: ProgressUpdate) => void;
  progressThrottleTime?: number;
  [key: string]: any;
}

/**
 * Interface for model loading options
 */
export interface ModelLoadOptions {
  model: string;
  quantization?: string;
  [key: string]: any;
}

/**
 * Interface for model initialization options
 */
export interface InitOptions {
  models?: string[];
  lazyLoad?: boolean;
  [key: string]: any;
}

/**
 * Interface for a chat message
 */
export interface ChatMessage {
  role: string;
  content: string;
  [key: string]: any;
}

/**
 * Interface for chat completion options
 */
export interface CompletionOptions {
  messages: ChatMessage[];
  stream?: boolean;
  streamOptions?: any;
  model?: string | null;
  temperature?: number;
  max_tokens?: number;
  do_sample?: boolean;
  top_k?: number;
  top_p?: number;
  [key: string]: any;
}

/**
 * Interface for a completion result
 */
export interface CompletionResult {
  id: string;
  object: string;
  created: number;
  model: string;
  choices: Array<{
    index: number;
    message: ChatMessage;
    finish_reason: string;
  }>;
  _tinylm?: {
    time_ms: number;
  };
}

/**
 * Interface for a streaming completion chunk
 */
export interface CompletionChunk {
  id: string;
  object: string;
  created: number;
  model: string;
  choices: Array<{
    index: number;
    delta: {
      content?: string;
      [key: string]: any;
    };
    finish_reason: string | null;
  }>;
}

/**
 * Interface for environment information
 */
export interface EnvInfo {
  backend?: string;
  cpuInfo?: string;
  gpuInfo?: string;
  [key: string]: any;
}

/**
 * Interface for capability information
 */
export interface CapabilityInfo extends WebGPUCapabilities {
  environment: EnvInfo;
  transformersVersion: string;
}

/**
 * Interface for the loaded model registry entry
 */
export interface ModelRegistryEntry {
  tokenizer: PreTrainedTokenizer;
  model: PreTrainedModel;
}

/**
 * Interface for the tokenized inputs
 */
export interface TokenizedInputs {
  input_ids: number[][];
  [key: string]: any;
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
            bytesLoaded: progress.progress || 0,
            bytesTotal: progress.total || 0
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
   * Offload a model from memory
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

      // Force garbage collection when possible
      if (typeof gc !== 'undefined') {
        gc();
      }

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

// Default export
export default TinyLM;
