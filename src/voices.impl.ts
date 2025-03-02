// Some of this code is from https://github.com/hexgrad/kokoro/

/**
 * Voice definitions for the TTS engine
 */
export const VOICES = Object.freeze({
  af: {
    // Default voice is a 50-50 mix of Bella & Sarah
    name: "Default",
    language: "en-us",
    gender: "Female",
  },
  af_bella: {
    name: "Bella",
    language: "en-us",
    gender: "Female",
  },
  af_nicole: {
    name: "Nicole",
    language: "en-us",
    gender: "Female",
  },
  af_sarah: {
    name: "Sarah",
    language: "en-us",
    gender: "Female",
  },
  af_sky: {
    name: "Sky",
    language: "en-us",
    gender: "Female",
  },
  am_adam: {
    name: "Adam",
    language: "en-us",
    gender: "Male",
  },
  am_michael: {
    name: "Michael",
    language: "en-us",
    gender: "Male",
  },

  bf_emma: {
    name: "Emma",
    language: "en-gb",
    gender: "Female",
  },
  bf_isabella: {
    name: "Isabella",
    language: "en-gb",
    gender: "Female",
  },
  bm_george: {
    name: "George",
    language: "en-gb",
    gender: "Male",
  },
  bm_lewis: {
    name: "Lewis",
    language: "en-gb",
    gender: "Male",
  },
  hf_alpha: {
    name: "alpha",
    language: "hi",
    gender: "Female",
  },
  hf_beta: {
    name: "beta",
    language: "hi",
    gender: "Female",
  },
  hm_omega: {
    name: "omega",
    language: "hi",
    gender: "Male",
  },
  hm_psi: {
    name: "psi",
    language: "hi",
    gender: "Male",
  },
  ef_dora: {
    name: "dora",
    language: "es",
    gender: "Female",
  },
  em_alex: {
    name: "alex",
    language: "es",
    gender: "Male",
  },
  em_santa: {
    name: "santa",
    language: "es",
    gender: "Male",
  },
  ff_siwis: {
    name: "siwis",
    language: "es",
    gender: "Female",
  },
  jf_alpha: {
    name: "alpha",
    language: "ja",
    gender: "Female",
  },
  jf_gongitsune: {
    name: "gongitsune",
    language: "ja",
    gender: "Female",
  },
  jf_nezumi: {
    name: "nezumi",
    language: "ja",
    gender: "Female",
  },
  jf_tebukuro: {
    name: "tebukuro",
    language: "ja",
    gender: "Female",
  },
  jm_kumo: {
    name: "kumo",
    language: "ja",
    gender: "Male",
  },
  zf_xiaobei: {
    name: "xiaobei",
    language: "zh",
    gender: "Female",
  },
  zf_xiaoni: {
    name: "xiaoni",
    language: "zh",
    gender: "Female",
  },
  zf_xiaoxiao: {
    name: "xiaoxiao",
    language: "zh",
    gender: "Female",
  },
  zf_xiaoyi: {
    name: "xiaoyi",
    language: "zh",
    gender: "Female",
  },
  zm_yunjian: {
    name: "yunjian",
    language: "zh",
    gender: "Male",
  },
  zm_yunxi: {
    name: "yunxi",
    language: "zh",
    gender: "Male",
  },
  zm_yunxia: {
    name: "yunxia",
    language: "zh",
    gender: "Male",
  },
  zm_yunyang: {
    name: "yunyang",
    language: "zh",
    gender: "Male",
  },
});

const VOICE_DATA_URL = "https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX/resolve/main/voices";

// Helper to detect environment
function isNodeEnvironment(): boolean {
  return typeof process !== 'undefined' &&
    process.versions != null &&
    process.versions.node != null;
}

// Polyfill for Headers in Node.js
class NodeHeaders {
  private headers: Record<string, string> = {};

  constructor(init?: Record<string, string>) {
    if (init) {
      Object.entries(init).forEach(([key, value]) => {
        this.headers[key.toLowerCase()] = value;
      });
    }
  }

  get(name: string): string | null {
    return this.headers[name.toLowerCase()] || null;
  }

  has(name: string): boolean {
    return name.toLowerCase() in this.headers;
  }

  // More methods can be added as needed
}

// Use native Headers or polyfill
const HeadersPolyfill = typeof Headers !== 'undefined' ? Headers : NodeHeaders;

// Polyfill for fetch in Node.js environments
async function safeFetch(url: string): Promise<{ arrayBuffer: () => Promise<ArrayBuffer>, headers: any }> {
  if (isNodeEnvironment()) {
    try {
      // Try native fetch first (Node.js v18+)
      if (typeof fetch === 'function') {
        return await fetch(url);
      }

      // Fallback to node-fetch or undici for older Node.js versions
      try {
        // Try to use undici (built into newer Node.js versions)
        const { fetch: undiciFetch } = await import('undici');
        return await undiciFetch(url);
      } catch (err) {
        // Fallback to using https module directly
        const https = await import('https');
        const { URL } = await import('url');

        const parsedUrl = new URL(url);

        return new Promise((resolve, reject) => {
          let data: Buffer[] = [];

          https.get({
            hostname: parsedUrl.hostname,
            path: parsedUrl.pathname + parsedUrl.search,
            headers: { 'User-Agent': 'TinyLM/1.0' }
          }, (res) => {
            if (res.statusCode !== 200) {
              reject(new Error(`Status Code: ${res.statusCode}`));
              return;
            }

            res.on('data', (chunk) => data.push(Buffer.from(chunk)));
            res.on('end', () => {
              const buffer = Buffer.concat(data);
              resolve({
                arrayBuffer: async () => buffer.buffer.slice(
                  buffer.byteOffset,
                  buffer.byteOffset + buffer.byteLength
                ),
                headers: new HeadersPolyfill(res.headers as Record<string, string>)
              });
            });
          }).on('error', reject);
        });
      }
    } catch (err) {
      console.error('Error using Node.js fetch methods:', err);
      throw err;
    }
  } else {
    // Browser environment - use native fetch
    return await fetch(url);
  }
}

/**
 * Fetch voice data from the voice file
 * @param {keyof typeof VOICES} id - Voice identifier
 * @returns {Promise<Float32Array>} Voice data
 */
async function getVoiceFile(id: string): Promise<ArrayBuffer> {
  const url = `${VOICE_DATA_URL}/${id}.bin`;

  // Detect if we're in Node.js environment
  const isNode = isNodeEnvironment();

  // Try to get from cache first
  if (isNode) {
    // Node.js environment - use fs for caching
    try {
      const fs = await import('fs/promises');
      const path = await import('path');
      const os = await import('os');

      // Create a cache directory in the system temp directory
      const cacheDir = path.join(os.tmpdir(), 'tinylm-voice-cache');

      try {
        await fs.mkdir(cacheDir, { recursive: true });
      } catch (err) {
        // Directory might already exist, ignore error
      }

      const cacheFilePath = path.join(cacheDir, `${id}.bin`);

      // Check if the file exists in cache
      try {
        const stats = await fs.stat(cacheFilePath);
        if (stats.isFile() && stats.size > 0) {
          // File exists in cache, read it
          const buffer = await fs.readFile(cacheFilePath);
          return buffer.buffer;
        }
      } catch (err) {
        // File doesn't exist, will fetch it
      }
    } catch (err) {
      // If there's any error with the file system, continue to network fetch
      console.warn("Unable to use file system cache:", err);
    }
  } else {
    // Browser environment - use Cache API if available
    if (typeof caches !== 'undefined') {
      try {
        const cache = await caches.open("kokoro-voices");
        const cachedResponse = await cache.match(url);
        if (cachedResponse) {
          return await cachedResponse.arrayBuffer();
        }
      } catch (e) {
        console.warn("Unable to use browser cache:", e);
      }
    }
  }

  // No cache hit, fetch the file
  const response = await safeFetch(url);
  const buffer = await response.arrayBuffer();

  // Cache the file for future use
  if (isNode) {
    try {
      const fs = await import('fs/promises');
      const path = await import('path');
      const os = await import('os');

      const cacheDir = path.join(os.tmpdir(), 'tinylm-voice-cache');
      const cacheFilePath = path.join(cacheDir, `${id}.bin`);

      // Write the file to cache
      await fs.writeFile(cacheFilePath, new Uint8Array(buffer));
    } catch (err) {
      console.warn("Unable to cache file in Node.js:", err);
    }
  } else if (typeof caches !== 'undefined' && typeof Response !== 'undefined') {
    // Browser caching only if Cache API and Response are available
    try {
      const cache = await caches.open("kokoro-voices");
      // NOTE: We use `new Response(buffer, ...)` instead of `response.clone()` to handle LFS files
      await cache.put(
        url,
        new Response(buffer, {
          headers: response.headers,
        }),
      );
    } catch (e) {
      console.warn("Unable to cache file in browser:", e);
    }
  }

  return buffer;
}

/**
 * Voice data cache
 */
const VOICE_CACHE = new Map<string, Float32Array>();

/**
 * Get voice data for a specific voice
 * @param {string} voice - Voice identifier
 * @returns {Promise<Float32Array>} Voice data
 */
export async function getVoiceData(voice: string): Promise<Float32Array> {
  if (VOICE_CACHE.has(voice)) {
    return VOICE_CACHE.get(voice)!;
  }

  const buffer = new Float32Array(await getVoiceFile(voice));
  VOICE_CACHE.set(voice, buffer);
  return buffer;
}
