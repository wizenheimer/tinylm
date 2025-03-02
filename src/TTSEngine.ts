/**
 * TTSEngine - Text-to-speech engine for TinyLM
 */

import { StyleTextToSpeech2Model, AutoTokenizer, Tensor, RawAudio } from "@huggingface/transformers";
import { phonemize } from './phonemize';
import { getVoiceData, VOICES } from './voices';
import { splitTextIntoSentences, ensureSafeTokenLength } from './TextSplitter';

const STYLE_DIM = 256;
const SAMPLE_RATE = 24000;

/**
 * Audio chunk with corresponding text
 */
export interface AudioChunk {
  text: string;
  audio: ArrayBuffer;
}

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
   * @returns {Promise<ArrayBuffer|AudioChunk[]>} Audio buffer or array of audio chunks if streaming
   */
  async generateSpeech(text: string, options: any = {}): Promise<ArrayBuffer | AudioChunk[]> {
    if (!this.model || !this.tokenizer) {
      throw new Error('TTS model not initialized');
    }

    const {
      voice = "af",
      speed = 1,
      stream = false, // New streaming parameter
    } = options;

    if (!VOICES.hasOwnProperty(voice)) {
      throw new Error(`Voice "${voice}" not found. Should be one of: ${Object.keys(VOICES).join(", ")}.`);
    }

    try {
      const language = voice.at(0); // "a" or "b"

      // If streaming is disabled, handle text as a single chunk (for backward compatibility)
      if (!stream) {
        // Process the entire text in a single call
        const phonemes = await phonemize(text, language);
        const { input_ids } = this.tokenizer(phonemes, { truncation: true });

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

        // Generate audio
        const { waveform } = await this.model(inputs);

        if (!waveform || !waveform.data) {
          throw new Error('Model returned null or undefined waveform');
        }

        // Use RawAudio to handle the audio data
        const rawAudio = new RawAudio(waveform.data, SAMPLE_RATE);
        return await rawAudio.toBlob().arrayBuffer();
      }
      else {
        // Streaming mode: Process text in chunks
        // Split text into sentences
        const sentences = splitTextIntoSentences(text);
        console.log(`Split text into ${sentences.length} sentences for streaming`);

        // If we have only one sentence, we might need to split it further
        const allChunks: string[] = [];
        for (const sentence of sentences) {
          const safeChunks = ensureSafeTokenLength(sentence);
          allChunks.push(...safeChunks);
        }

        // Process each chunk
        const audioChunks: AudioChunk[] = [];

        for (const chunk of allChunks) {
          if (!chunk.trim()) continue; // Skip empty chunks

          // Generate phonemes for this chunk
          const phonemes = await phonemize(chunk, language);
          const { input_ids } = this.tokenizer(phonemes, { truncation: true });

          // Select voice style based on number of input tokens
          const num_tokens = Math.min(Math.max(
            input_ids.dims.at(-1) - 2,
            0,
          ), 509);

          // Load voice style
          const data = await getVoiceData(voice);
          const offset = num_tokens * STYLE_DIM;
          const voiceData = data.slice(offset, offset + STYLE_DIM);

          // Prepare inputs
          const inputs = {
            input_ids,
            style: new Tensor("float32", voiceData, [1, STYLE_DIM]),
            speed: new Tensor("float32", [speed], [1]),
          };

          // Generate audio for this chunk
          const { waveform } = await this.model(inputs);

          if (!waveform || !waveform.data) {
            console.warn('Model returned null or undefined waveform for chunk:', chunk);
            continue;
          }

          // Create audio buffer for this chunk
          const rawAudio = new RawAudio(waveform.data, SAMPLE_RATE);
          const audioBuffer = await rawAudio.toBlob().arrayBuffer();

          // Add to results
          audioChunks.push({
            text: chunk,
            audio: audioBuffer
          });
        }

        return audioChunks;
      }
    } catch (error) {
      console.error('Error in generateSpeech:', error);
      throw error;
    }
  }

  /**
   * Generate combined audio from multiple sentence chunks
   * @param {string[]} sentences - Array of sentences to process
   * @param {string} voice - Voice ID
   * @param {number} speed - Speech speed
   * @returns {Promise<ArrayBuffer>} Combined audio buffer
   */
  async generateCombinedSpeech(sentences: string[], voice: string, speed: number): Promise<ArrayBuffer> {
    if (!this.model || !this.tokenizer) {
      throw new Error('TTS model not initialized');
    }

    try {
      const language = voice.at(0);
      const audioChunks: Float32Array[] = [];
      let totalLength = 0;

      // Short pause between sentences (200ms)
      const PAUSE_DURATION = 0.2;
      const pauseSamples = Math.floor(SAMPLE_RATE * PAUSE_DURATION);
      const pauseData = new Float32Array(pauseSamples).fill(0);

      for (const sentence of sentences) {
        if (!sentence.trim()) continue;

        // Process this sentence
        const phonemes = await phonemize(sentence, language);
        const { input_ids } = this.tokenizer(phonemes, { truncation: true });

        // Select voice style based on number of input tokens
        const num_tokens = Math.min(Math.max(input_ids.dims.at(-1) - 2, 0), 509);

        // Load voice style
        const data = await getVoiceData(voice);
        const offset = num_tokens * STYLE_DIM;
        const voiceData = data.slice(offset, offset + STYLE_DIM);

        // Generate audio
        const inputs = {
          input_ids,
          style: new Tensor("float32", voiceData, [1, STYLE_DIM]),
          speed: new Tensor("float32", [speed], [1]),
        };

        const { waveform } = await this.model(inputs);

        if (waveform && waveform.data) {
          // Add audio data
          const sentenceAudio = new Float32Array(waveform.data);
          audioChunks.push(sentenceAudio);
          totalLength += sentenceAudio.length;

          // Add pause after each sentence (except the last one)
          if (sentence !== sentences[sentences.length - 1]) {
            audioChunks.push(pauseData);
            totalLength += pauseData.length;
          }
        }
      }

      // Combine all audio chunks
      const combinedAudio = new Float32Array(totalLength);
      let position = 0;

      for (const chunk of audioChunks) {
        combinedAudio.set(chunk, position);
        position += chunk.length;
      }

      // Create final audio
      const rawAudio = new RawAudio(combinedAudio, SAMPLE_RATE);
      return await rawAudio.toBlob().arrayBuffer();

    } catch (error) {
      console.error('Error in generateCombinedSpeech:', error);
      throw error;
    }
  }
}
