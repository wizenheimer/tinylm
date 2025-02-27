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
