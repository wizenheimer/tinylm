/**
 * TinyLM Utility Functions
 */

// Declare a utility type for garbage collection without conflicting with Node types
type GarbageCollectionFunction = () => void;

/**
 * Environment detection result
 */
export interface EnvironmentInfo {
  isNode: boolean;
  isBrowser: boolean;
  isWebWorker: boolean;
  isDeno: boolean;
  isEdgeFunction: boolean;
  isCloudflareWorker: boolean;
}

/**
 * Detect the current JavaScript runtime environment
 * @returns {EnvironmentInfo} Environment information
 */
export function detectEnvironment(): EnvironmentInfo {
  const env: EnvironmentInfo = {
    isNode: false,
    isBrowser: false,
    isWebWorker: false,
    isDeno: false,
    isEdgeFunction: false,
    isCloudflareWorker: false
  };

  // Check for Node.js
  if (typeof process !== 'undefined' &&
      process.versions != null &&
      process.versions.node != null) {
    env.isNode = true;
    return env;
  }

  // Check for browser
  if (typeof window !== 'undefined' && typeof document !== 'undefined') {
    env.isBrowser = true;
    return env;
  }

  // Check for Web Worker
  if (typeof self !== 'undefined' && typeof self.importScripts === 'function') {
    env.isWebWorker = true;
    return env;
  }
  // Check for Deno
  if (typeof globalThis !== 'undefined' && 'Deno' in globalThis) {
    env.isDeno = true;
    return env;
  }

  // Check for Cloudflare Workers
  if (typeof self !== 'undefined' && typeof caches !== 'undefined') {
    env.isCloudflareWorker = true;
    return env;
  }

  // Check for Edge Function (Vercel, Netlify)
  if (typeof process !== 'undefined' && process.env && process.env.EDGE_RUNTIME) {
    env.isEdgeFunction = true;
    return env;
  }

  // Default to browser-like environment if we can't determine
  env.isBrowser = true;
  return env;
}

/**
 * Cross-platform utility to safely trigger garbage collection if available
 * Works in Node.js (with --expose-gc flag) and browsers (if they expose gc)
 */
export function tryGarbageCollection(): void {
  try {
    // Try to access gc in the current environment
    // Browser: globalThis.gc
    // Node.js: global.gc
    // Note: TypeScript will complain about gc not existing on globalThis
    // but we handle that with the try-catch
    const gc = (globalThis as any).gc;

    if (typeof gc === 'function') {
      gc();
    }
  } catch (error) {
    // Silently ignore - GC not available
    console.debug('Garbage collection not available in this environment');
  }
}


/**
 * Log audio generation debug info
 * @param {any} audioData - Audio data for debugging
 * @param {string} stage - Processing stage
 */
export function logAudioDebug(audioData: any, stage: string = 'unknown') {
  if (!audioData) {
    console.warn(`[Audio Debug] ${stage}: No audio data`);
    return;
  }

  const length = audioData.length || (audioData.data ? audioData.data.length : 0);
  const sampleValues = length > 0 ? Array.from(audioData.data || audioData).slice(0, 5) : [];
  const hasNaN = Array.from(audioData.data || audioData).some((v: any) => isNaN(v));

  const stats = {
    stage,
    length,
    hasValues: length > 0,
    hasNonZero: Array.from(audioData.data || audioData).some((v: any) => v !== 0),
    sampleValues,
    hasNaN,
    type: audioData.constructor.name
  };

  console.log("[Audio Debug]", stats);
}

/**
 * Safely analyze audio data without causing stack overflows
 */
export function analyzeAudioData(audioData: Float32Array | Int16Array | null, label: string = "Unknown"): void {
  if (!audioData || audioData.length === 0) {
    console.warn(`[Audio Analysis] ${label}: No audio data or empty array`);
    return;
  }

  // Sample size instead of analyzing entire array
  const MAX_ANALYSIS_SIZE = 1000;
  const dataLength = audioData.length;

  // Calculate min, max, and average iteratively
  let min = Infinity;
  let max = -Infinity;
  let sum = 0;
  let nanCount = 0;
  let silenceCount = 0;

  // Analyze only first and last portions to avoid memory issues
  const samplesToAnalyze = Math.min(dataLength, MAX_ANALYSIS_SIZE);
  const firstHalf = Math.floor(samplesToAnalyze / 2);

  // Analyze beginning of array
  for (let i = 0; i < firstHalf; i++) {
    if (i >= 0 && i < dataLength) {
      const val = audioData[i];
      if (val !== undefined) {
        if (isNaN(val)) {
          nanCount++;
        } else {
          min = Math.min(min, val);
          max = Math.max(max, val);
          sum += val;
          if (Math.abs(val) < 0.01) silenceCount++;
        }
      }
    }
  }

  // Analyze end of array
  for (let i = Math.max(0, dataLength - firstHalf); i < dataLength; i++) {
    if (i >= 0 && i < dataLength) {
      const val = audioData[i];
      if (val !== undefined) {
        if (isNaN(val)) {
          nanCount++;
        } else {
          min = Math.min(min, val);
          max = Math.max(max, val);
          sum += val;
          if (Math.abs(val) < 0.01) silenceCount++;
        }
      }
    }
  }

  // Guard against division by zero
  const avg = samplesToAnalyze > 0 ? sum / samplesToAnalyze : 0;

  // Get samples without using slice or map
  const firstSamples: string[] = [];
  const lastSamples: string[] = [];

  for (let i = 0; i < 5 && i < dataLength; i++) {
    const value = audioData[i];
    if (value !== undefined) {
      firstSamples.push(Number(value).toFixed(4));
    }
  }

  for (let i = Math.max(0, dataLength - 5); i < dataLength; i++) {
    const value = audioData[i];
    if (value !== undefined) {
      lastSamples.push(Number(value).toFixed(4));
    }
  }

  console.log(`[Audio Analysis] ${label} (${dataLength} samples):
    - Range: ${min.toFixed(4)} to ${max.toFixed(4)}
    - Average: ${avg.toFixed(4)}
    - NaN values: ${nanCount}
    - Mostly silent: ${(silenceCount / samplesToAnalyze) > 0.9}
    - First ${firstSamples.length} samples: [${firstSamples.join(', ')}]
    - Last ${lastSamples.length} samples: [${lastSamples.join(', ')}]
  `);
}
