/**
 * TextStreamHandler - A simple wrapper to handle streaming text generation
 * Avoids direct references to TextStreamer .write method which causes the issue
 */

import { PreTrainedTokenizer } from "@huggingface/transformers";

/**
 * Create a streamer that works in all environments
 * @param {PreTrainedTokenizer} tokenizer - The tokenizer for decoding
 * @param {Object} options - Streamer options
 * @returns {Object} A streamer object compatible with the generate method
 */
export async function createStreamer(tokenizer: PreTrainedTokenizer, options: {
  skip_prompt?: boolean;
  skip_special_tokens?: boolean;
  callback?: (text: string) => void;
}) {
  // Dynamically import TextStreamer to avoid reference errors
  // This is the key to making it work in different environments
  const { TextStreamer } = await import("@huggingface/transformers");

  // Create the streamer with the callback built in
  return new TextStreamer(tokenizer, {
    skip_prompt: options.skip_prompt !== false,
    skip_special_tokens: options.skip_special_tokens !== false,
    callback_function: options.callback || (() => {})
  });
}
