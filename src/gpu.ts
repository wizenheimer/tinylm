import { DeviceConfig, WebGPUCapabilities } from "./types";

export class WebGPUChecker {
  private isWebGPUSupported: boolean = false;
  private fp16Supported: boolean = false;
  private adapter: GPUAdapter | null = null;

  /**
   * Check if WebGPU is supported in the current environment
   * @returns {Promise<WebGPUCapabilities>} WebGPU capabilities
   */
  async check(): Promise<WebGPUCapabilities> {
    try {
      // Check for Node.js environment safely
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

      // Check for browser WebGPU support safely
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
