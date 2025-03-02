/**
 * TTSEngine - Text-to-speech engine for TinyLM
 * Rewritten to match KokoroJS exactly
 */

import { StyleTextToSpeech2Model, AutoTokenizer, Tensor, RawAudio } from "@huggingface/transformers";
import { phonemize } from './phonemize';
import { getVoiceData, VOICES } from './voices';
import { splitTextIntoSentences, ensureSafeTokenLength } from './TextSplitter';
import { analyzeAudioData } from './utils';

// FIXED: Ensure constants match KokoroJS exactly
const STYLE_DIM = 256;
const SAMPLE_RATE = 24000;

/**
 * Audio chunk with corresponding text
 */
export interface AudioChunk {
  text: string;
  audio: ArrayBuffer;
  phonemes?: string;
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

      console.log("TTS model and tokenizer loaded successfully");
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

    // Validate voice exists
    if (!VOICES.hasOwnProperty(voice)) {
      throw new Error(`Voice "${voice}" not found. Should be one of: ${Object.keys(VOICES).join(", ")}.`);
    }

    try {
      // Get language code from voice identifier (first letter)
      const language = voice.substring(0, 1) as 'a' | 'b' | 'h' | 'e' | 'f' | 'z';
      console.log(`Using voice '${voice}' with language code '${language}'`);

      // If streaming is disabled and text is short, handle as a single chunk
      if (!stream && text.length < 500) {
        console.log("Processing as single chunk (stream=false)");

        try {
          // Process the entire text in a single call
          const phonemes = await phonemize(text, language);
          console.log(`Phonemized text: "${phonemes.substring(0, 50)}${phonemes.length > 50 ? '...' : ''}"`);

          const tokenizeResult = this.tokenizer(phonemes, { truncation: true });
          const input_ids = tokenizeResult.input_ids;

          console.log(`Tokenized to ${input_ids.dims.at(-1)} tokens`);

          // Select voice style based on number of input tokens
          const num_tokens = Math.min(Math.max(
            input_ids.dims.at(-1) - 2, // Without padding
            0,
          ), 509);

          console.log(`Using ${num_tokens} tokens for voice style selection`);

          // Load voice style
          const data = await getVoiceData(voice);
          const offset = num_tokens * STYLE_DIM;

          if (offset + STYLE_DIM > data.length) {
            console.warn(`Warning: Offset ${offset} exceeds voice data length ${data.length}, using fallback`);
          }

          const voiceData = offset + STYLE_DIM <= data.length
            ? data.slice(offset, offset + STYLE_DIM)
            : data.slice(0, STYLE_DIM); // Fallback to first style

          // Prepare model inputs exactly like KokoroJS
          const inputs = {
            input_ids,
            style: new Tensor("float32", voiceData, [1, STYLE_DIM]),
            speed: new Tensor("float32", [speed], [1]),
          };

          console.log("Generating audio...");

          // Generate audio - use direct model call like KokoroJS
          const output = await this.model(inputs);

          if (!output || !output.waveform || !output.waveform.data) {
            throw new Error('Model returned null or undefined waveform');
          }


          // Skip audio analysis in normal operation to avoid stack overflow
          // Only uncomment when debugging
          try {
            if (output.waveform.data.length < 50000) { // Only analyze smaller arrays
              analyzeAudioData(output.waveform.data, "Model Output");
            } else {
              console.log(`Skipping analysis of large waveform (${output.waveform.data.length} samples)`);
            }
          } catch (analyzeError) {
            console.warn("Error analyzing audio data:", analyzeError);
          }

          // Use RawAudio to handle conversion properly
          const rawAudio = new RawAudio(output.waveform.data, SAMPLE_RATE);

          // Convert to blob/buffer and return
          return await rawAudio.toBlob().arrayBuffer();

        } catch (error) {
          console.error("Error in single-chunk processing:", error);
          throw error;
        }
      }
      else {
        // Streaming mode: Process text in chunks
        console.log("Processing with streaming enabled or text is long");

        // Split text into sentences
        const sentences = splitTextIntoSentences(text);
        console.log(`Split text into ${sentences.length} sentences for streaming`);

        // If we have only one sentence, we might need to split it further
        const allChunks: string[] = [];
        for (const sentence of sentences) {
          const safeChunks = ensureSafeTokenLength(sentence);
          allChunks.push(...safeChunks);
        }

        console.log(`Processing ${allChunks.length} total chunks`);

        // Process each chunk
        const audioChunks: AudioChunk[] = [];

        for (const chunk of allChunks) {
          if (!chunk.trim()) continue; // Skip empty chunks

          try {
            // Generate phonemes for this chunk
            const phonemes = await phonemize(chunk, language);
            const tokenizeResult = this.tokenizer(phonemes, { truncation: true });
            const input_ids = tokenizeResult.input_ids;

            // Select voice style based on number of input tokens
            const num_tokens = Math.min(Math.max(
              input_ids.dims.at(-1) - 2,
              0,
            ), 509);

            // Load voice style
            const data = await getVoiceData(voice);
            const offset = num_tokens * STYLE_DIM;

            const voiceData = offset + STYLE_DIM <= data.length
              ? data.slice(offset, offset + STYLE_DIM)
              : data.slice(0, STYLE_DIM); // Fallback

            // Prepare inputs
            const inputs = {
              input_ids,
              style: new Tensor("float32", voiceData, [1, STYLE_DIM]),
              speed: new Tensor("float32", [speed], [1]),
            };

            // Generate audio for this chunk
            const output = await this.model(inputs);

            if (!output || !output.waveform || !output.waveform.data) {
              console.warn('Model returned null or undefined waveform for chunk:', chunk);
              continue;
            }

            // Create audio buffer for this chunk
            const rawAudio = new RawAudio(output.waveform.data, SAMPLE_RATE);
            const audioBuffer = await rawAudio.toBlob().arrayBuffer();

            // Add to results
            audioChunks.push({
              text: chunk,
              audio: audioBuffer,
              phonemes
            });

          } catch (chunkError) {
            console.error(`Error processing chunk "${chunk}":`, chunkError);
            // Continue with other chunks
          }
        }

        return audioChunks;
      }
    } catch (error) {
      console.error('Error in generateSpeech:', error);
      throw new Error(`Speech generation failed: ${error instanceof Error ? error.message : String(error)}`);
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
      const language = voice.substring(0, 1);
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
        const tokenizeResult = this.tokenizer(phonemes, { truncation: true });
        const input_ids = tokenizeResult.input_ids;

        // Select voice style based on number of input tokens
        const num_tokens = Math.min(Math.max(input_ids.dims.at(-1) - 2, 0), 509);

        // Load voice style
        const data = await getVoiceData(voice);
        const offset = num_tokens * STYLE_DIM;

        const voiceData = offset + STYLE_DIM <= data.length
          ? data.slice(offset, offset + STYLE_DIM)
          : data.slice(0, STYLE_DIM); // Fallback

        // Generate audio
        const inputs = {
          input_ids,
          style: new Tensor("float32", voiceData, [1, STYLE_DIM]),
          speed: new Tensor("float32", [speed], [1]),
        };

        const output = await this.model(inputs);

        if (output && output.waveform && output.waveform.data) {
          // Add audio data
          const sentenceAudio = new Float32Array(output.waveform.data);
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

  /**
   * Validate voice is available and properly formatted
   * @private
   * @param {string} voice - Voice ID to validate
   * @returns {string} Language code
   */
  private _validateVoice(voice: string): string {
    if (!VOICES.hasOwnProperty(voice)) {
      console.error(`Voice "${voice}" not found. Available voices:`, Object.keys(VOICES));
      throw new Error(`Voice "${voice}" not found. Should be one of: ${Object.keys(VOICES).join(", ")}.`);
    }

    const language = voice.substring(0, 1);
    return language;
  }
}
