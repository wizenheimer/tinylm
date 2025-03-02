/**
 * Voice data utilities for TinyLM
 * Improved to match KokoroJS with better validation
 */

/**
 * Voice definitions for the TTS engine
 * Exact copy of KokoroJS voice definitions
 */
export const VOICES = Object.freeze({
  af: {
    // Default voice is a 50-50 mix of Bella & Sarah
    name: "Default",
    language: "en-us",
    gender: "Female",
  },
  af_heart: {
    name: "Heart",
    language: "en-us",
    gender: "Female",
    traits: "❤️",
    targetQuality: "A",
    overallGrade: "A",
  },
  af_bella: {
    name: "Bella",
    language: "en-us",
    gender: "Female",
    traits: "🔥",
    targetQuality: "A",
    overallGrade: "A-",
  },
  af_nicole: {
    name: "Nicole",
    language: "en-us",
    gender: "Female",
    traits: "🎧",
    targetQuality: "B",
    overallGrade: "B-",
  },
  af_sarah: {
    name: "Sarah",
    language: "en-us",
    gender: "Female",
    targetQuality: "B",
    overallGrade: "C+",
  },
  af_sky: {
    name: "Sky",
    language: "en-us",
    gender: "Female",
    targetQuality: "B",
    overallGrade: "C-",
  },
  am_adam: {
    name: "Adam",
    language: "en-us",
    gender: "Male",
    targetQuality: "D",
    overallGrade: "F+",
  },
  am_michael: {
    name: "Michael",
    language: "en-us",
    gender: "Male",
    targetQuality: "B",
    overallGrade: "C+",
  },
  bf_emma: {
    name: "Emma",
    language: "en-gb",
    gender: "Female",
    traits: "🚺",
    targetQuality: "B",
    overallGrade: "B-",
  },
  bf_isabella: {
    name: "Isabella",
    language: "en-gb",
    gender: "Female",
    targetQuality: "B",
    overallGrade: "C",
  },
  bm_george: {
    name: "George",
    language: "en-gb",
    gender: "Male",
    targetQuality: "B",
    overallGrade: "C",
  },
  bm_lewis: {
    name: "Lewis",
    language: "en-gb",
    gender: "Male",
    targetQuality: "C",
    overallGrade: "D+",
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
});

/**
 * Voice data cache
 */
const VOICE_CACHE = new Map<string, Float32Array>();

// Match KokoroJS URL exactly
const VOICE_DATA_URL = "https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX/resolve/main/voices";

// Helper to detect environment
function isNodeEnvironment(): boolean {
  return typeof process !== 'undefined' &&
    process.versions != null &&
    process.versions.node != null;
}

/**
 * Fetch voice data from the voice file
 * @param {string} id - Voice identifier
 * @returns {Promise<ArrayBuffer>} Voice data
 */
async function getVoiceFile(id: string): Promise<ArrayBuffer> {
  // Create the URL exactly like KokoroJS
  const url = `${VOICE_DATA_URL}/${id}.bin`;
  console.log(`Fetching voice file from: ${url}`);

  // Browser environment - use Cache API if available
  if (!isNodeEnvironment() && typeof caches !== 'undefined') {
    try {
      const cache = await caches.open("kokoro-voices");
      const cachedResponse = await cache.match(url);
      if (cachedResponse) {
        console.log(`Using cached voice data for ${id}`);
        return await cachedResponse.arrayBuffer();
      }
    } catch (e) {
      console.warn("Unable to use browser cache:", e);
    }
  }

  // No cache hit, fetch the file
  console.log(`Downloading voice data for ${id}...`);
  try {
    const response = await fetch(url);

    if (!response.ok) {
      throw new Error(`Failed to fetch voice data: ${response.status} ${response.statusText}`);
    }

    const buffer = await response.arrayBuffer();

    if (!buffer || buffer.byteLength === 0) {
      throw new Error(`Downloaded voice file is empty`);
    }

    console.log(`Successfully downloaded ${buffer.byteLength} bytes for voice ${id}`);

    // Cache the file for future use
    if (!isNodeEnvironment() && typeof caches !== 'undefined' && typeof Response !== 'undefined') {
      try {
        const cache = await caches.open("kokoro-voices");
        await cache.put(url, new Response(buffer.slice(0), {
          headers: response.headers,
        }));
        console.log(`Cached voice data for ${id}`);
      } catch (e) {
        console.warn("Unable to cache file in browser:", e);
      }
    }

    return buffer;
  } catch (error) {
    console.error(`Error fetching voice file for ${id}:`, error);
    throw new Error(`Failed to download voice ${id}: ${error}`);
  }
}

/**
 * Get voice data for a specific voice
 * @param {string} voice - Voice identifier
 * @returns {Promise<Float32Array>} Voice data
 */
export async function getVoiceData(voice: string): Promise<Float32Array> {
  // Use cached data if available
  if (VOICE_CACHE.has(voice)) {
    const cached = VOICE_CACHE.get(voice);
    if (cached && cached.length > 0) {
      console.log(`Using cached voice data for ${voice}`);
      return cached;
    }

    // Clear invalid cache entry
    console.warn(`Cached voice data for ${voice} was invalid, re-downloading`);
    VOICE_CACHE.delete(voice);
  }

  console.log(`Loading voice data for '${voice}'...`);

  // Validate voice is in our registry
  if (!VOICES.hasOwnProperty(voice)) {
    console.error(`Voice "${voice}" not found. Available voices: ${Object.keys(VOICES).join(", ")}`);
    throw new Error(`Voice "${voice}" not found.`);
  }

  try {
    // Get the binary file
    const buffer = await getVoiceFile(voice);

    if (!buffer || buffer.byteLength === 0) {
      throw new Error(`Voice file for '${voice}' is empty or invalid`);
    }

    // Create Float32Array with validation
    const data = new Float32Array(buffer);

    if (data.length === 0) {
      throw new Error(`Voice data conversion resulted in empty array`);
    }

    // Verify the data isn't all zeros
    const hasNonZero = Array.from(data.slice(0, 100)).some(val => val !== 0);
    if (!hasNonZero) {
      console.warn(`Warning: First 100 values of voice data for '${voice}' are all zeros`);
    }

    // Cache and return
    VOICE_CACHE.set(voice, data);

    // Log some data statistics for debugging
    const min = Math.min(...Array.from(data.slice(0, 1000)));
    const max = Math.max(...Array.from(data.slice(0, 1000)));
    console.log(`Voice data loaded for ${voice}: ${data.length} samples, range in first 1000: [${min}, ${max}]`);

    return data;
  } catch (error) {
    console.error(`Error loading voice data for '${voice}':`, error);
    throw new Error(`Failed to load voice data for '${voice}': ${error instanceof Error ? error.message : String(error)}`);
  }
}
