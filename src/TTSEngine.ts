/**
 * TTSEngine - Text-to-speech engine for TinyLM
 */

import { StyleTextToSpeech2Model, AutoTokenizer, Tensor, RawAudio } from "@huggingface/transformers";
import { phonemize } from './phonemize';
import { getVoiceData, VOICES } from './voices';

const STYLE_DIM = 256;
const SAMPLE_RATE = 24000;

/**
 * Text-to-speech engine for TinyLM
 */
export class TTSEngine {
  private model: any = null;
  private tokenizer: any = null;

  /**
   * Create a new TTS engine
   */
  constructor() {
    this.model = null;
    this.tokenizer = null;
  }

  /**
   * Load a TTS model
   * @param {string} model - Model name
   * @param {any} options - Loading options
   * @returns {Promise<void>} Promise that resolves when the model is loaded
   */
  async loadModel(model: string, options: any = {}): Promise<void> {
    try {
      console.log(`Loading TTS model with device="${options.device || 'wasm'}", dtype="${options.dtype || 'fp32'}"`);

      this.model = await StyleTextToSpeech2Model.from_pretrained(model, {
        progress_callback: options.onProgress,
        dtype: options.dtype || "fp32",
        device: options.device || "wasm", // Use WASM as default fallback instead of CPU
        // TODO: maybe have "cpu" for node runtime and "wasm" for web as a fallback
      });

      this.tokenizer = await AutoTokenizer.from_pretrained(model, {
        progress_callback: options.onProgress
      });
    } catch (error) {
      console.error('Error loading TTS model:', error);
      throw error;
    }
  }

  /**
   * Generate speech from text
   * @param {string} text - Input text
   * @param {any} options - Generation options
   * @returns {Promise<ArrayBuffer>} Audio buffer
   */
  async generateSpeech(text: string, options: any = {}): Promise<ArrayBuffer> {
    if (!this.model || !this.tokenizer) {
      throw new Error('TTS model not initialized');
    }

    const { voice = "af", speed = 1 } = options;

    if (!VOICES.hasOwnProperty(voice)) {
      throw new Error(`Voice "${voice}" not found. Should be one of: ${Object.keys(VOICES).join(", ")}.`);
    }

    try {
      const language = voice.at(0); // "a" or "b"
      const phonemes = await phonemize(text, language);

      const { input_ids } = this.tokenizer(phonemes, {
        truncation: true,
      });

      // Select voice style based on number of input tokens
      const num_tokens = Math.min(Math.max(
        input_ids.dims.at(-1) - 2, // Without padding
        0,
      ), 509);

      // Load voice style
      const data = await getVoiceData(voice);
      const offset = num_tokens * STYLE_DIM;
      const voiceData = data.slice(offset, offset + STYLE_DIM);

      // Prepare model inputs
      const inputs = {
        input_ids,
        style: new Tensor("float32", voiceData, [1, STYLE_DIM]),
        speed: new Tensor("float32", [speed], [1]),
      };

      // Generate audio - use direct model call like KokoroJS
      const { waveform } = await this.model(inputs);

      if (!waveform || !waveform.data) {
        throw new Error('Model returned null or undefined waveform');
      }

      // Use RawAudio to properly handle the audio data (similar to KokoroJS)
      const rawAudio = new RawAudio(waveform.data, SAMPLE_RATE);

      // Convert to blob and return as ArrayBuffer
      return await rawAudio.toBlob().arrayBuffer();
    } catch (error) {
      console.error('Error in generateSpeech:', error);
      throw error;
    }
  }
}
