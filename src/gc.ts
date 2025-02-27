/**
 * TinyLM - Cross-platform garbage collection utility
 */

// Declare a utility type for garbage collection without conflicting with Node types
type GarbageCollectionFunction = () => void;

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
