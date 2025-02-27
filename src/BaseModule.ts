/**
 * Base module interface for TinyLM features
 * Provides common structure for all modules like generation, embeddings, etc.
 */

import { WebGPUChecker } from './WebGPUChecker';
import { ProgressTracker } from './ProgressTracker';
import { tryGarbageCollection, EnvironmentInfo } from './utils';
import { DeviceConfig } from './types';

/**
 * Base interface for TinyLM modules
 */
export interface TinyLMModule {
  /**
   * Initialize the module
   * @param {InitOptions} options - Initialization options
   * @returns {Promise<void>} Initialization result
   */
  init(options?: Record<string, any>): Promise<void>;
}

/**
 * Base class for TinyLM modules with common functionality
 */
export abstract class BaseModule implements TinyLMModule {
  protected tinyLM: any;
  protected progressTracker: ProgressTracker;
  protected webgpuChecker: WebGPUChecker;
  protected environment: EnvironmentInfo;

  /**
   * Create a new module
   * @param {any} tinyLM - Parent TinyLM instance
   */
  constructor(tinyLM: any) {
    this.tinyLM = tinyLM;
    this.progressTracker = tinyLM.getProgressTracker();
    this.webgpuChecker = tinyLM.getWebGPUChecker();
    this.environment = tinyLM.getEnvironment();
  }

  /**
   * Initialize the module
   * @param {Record<string, any>} options - Initialization options
   * @returns {Promise<void>} Initialization result
   */
  async init(options: Record<string, any> = {}): Promise<void> {
    // Base initialization code
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
      // Use the utility function
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
}
