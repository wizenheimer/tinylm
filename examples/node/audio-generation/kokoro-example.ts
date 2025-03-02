/**
 * Text-to-Speech Example with TinyLM
 *
 * This demonstrates speech generation with TinyLM, including:
 * - Basic speech generation
 * - Streaming for better handling of long texts
 * - Comparing streaming vs non-streaming approaches
 */

import { TinyLM, ProgressUpdate, SpeechResult, SpeechStreamResult } from '../../../src/index';
import * as fs from 'fs';
import * as path from 'path';
import { fileURLToPath } from 'url';

// Get current directory (ES module compatible)
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Format progress for console output
function formatProgress(progress: ProgressUpdate): string {
  const { type, status, percentComplete, message } = progress;

  // Progress bar for numeric progress
  let progressBar = '';
  if (typeof percentComplete === 'number' && percentComplete >= 0 && percentComplete <= 100) {
    const barWidth = 20;
    const filledWidth = Math.round((barWidth * percentComplete) / 100);
    progressBar = '[' +
      '#'.repeat(filledWidth) +
      '-'.repeat(barWidth - filledWidth) +
      `] ${percentComplete}%`;
  }

  // Format output lines
  let output = `[${status}] ${type ? `(${type})` : ''}`;
  if (progressBar) output += ` ${progressBar}`;
  if (message) output += ` ${message}`;

  return output;
}

// Create an output directory for audio files
function ensureOutputDirExists(): string {
  const outputDir = path.join(__dirname, 'audio-output');
  if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
  }
  return outputDir;
}

// Type guard to check if result is a streaming result
function isStreamResult(result: SpeechResult | SpeechStreamResult): result is SpeechStreamResult {
  return 'chunks' in result && Array.isArray((result as SpeechStreamResult).chunks);
}

// Main text-to-speech example
async function runTextToSpeechExample(): Promise<void> {
  console.log('=== TinyLM Text-to-Speech Example ===');

  // Create an output directory
  const outputDir = ensureOutputDirExists();

  // Create a new TinyLM instance with custom progress tracking
  const tiny = new TinyLM({
    progressCallback: (progress: ProgressUpdate) => {
      console.log(formatProgress(progress));
    },
    progressThrottleTime: 100
  });

  try {
    // Check hardware capabilities first
    console.log("\nChecking hardware capabilities...");
    const capabilities = await tiny.models.check();
    console.log("WebGPU available:", capabilities.isWebGPUSupported);
    console.log("FP16 supported:", capabilities.fp16Supported);
    if (capabilities.environment) {
      console.log("Backend:", capabilities.environment.backend);
    }

    // Initialize TinyLM
    console.log("\nInitializing TinyLM...");
    await tiny.init({
      ttsModels: ['onnx-community/Kokoro-82M-v1.0-ONNX']
    });

    // Example 1: Basic speech generation (non-streaming)
    console.log("\n=== Example 1: Basic Speech Generation ===");
    const shortText = "Welcome to TinyLM. This is a library for running language models and text-to-speech in browsers and Node.js.";
    console.log(`\nGenerating speech for: "${shortText}"`);

    const basicResult = await tiny.audio.speech.create({
      model: 'onnx-community/Kokoro-82M-v1.0-ONNX',
      input: shortText,
      voice: 'af_bella',
      response_format: 'wav'
    });

    // Output result is a regular SpeechResult
    if (!isStreamResult(basicResult)) {
      const basicPath = path.join(outputDir, 'basic_speech.wav');
      fs.writeFileSync(basicPath, Buffer.from(basicResult.audio));
      console.log(`Speech saved to: ${basicPath}`);
      console.log(`Generation time: ${basicResult._tinylm?.time_ms}ms`);
    }

    // Example 2: Streaming speech generation for long text
    console.log("\n=== Example 2: Streaming TTS for Long Text ===");
    const longText = `
    Streaming text-to-speech processes content in semantically meaningful chunks.
    This creates more natural speech with proper phrasing and intonation.
    Unlike non-streaming approaches, this maintains consistent prosody across sentence boundaries.
    The implementation handles sentence boundaries, ensuring natural pauses between thoughts.
    It's particularly useful for longer texts like articles or stories.
    When texts are processed as a whole, long content can lose natural cadence and timing.
    Streaming solves this by breaking content into manageable pieces.
    Each piece receives appropriate voice styling based on its content and length.
    The result is more human-like speech that's easier to follow and understand.
    `;

    console.log(`\nGenerating streaming speech for long text (${longText.length} characters)`);

    // Generate speech with streaming enabled
    const streamResult = await tiny.audio.speech.create({
      model: 'onnx-community/Kokoro-82M-v1.0-ONNX',
      input: longText,
      voice: 'af_bella',
      response_format: 'wav',
      stream: true // Enable streaming
    });

    // Check if result is a streaming result
    if (isStreamResult(streamResult)) {
      // Instead of trying to concatenate the chunks:
      console.log("\nNOTE: Streaming mode produces multiple audio files - one per chunk.");
      console.log("For production use, consider using a proper audio library for concatenation.");

      // Save each chunk separately
      for (let i = 0; i < streamResult.chunks.length; i++) {
        const chunk = streamResult.chunks[i];
        if (chunk) {
          const chunkPath = path.join(outputDir, `stream_chunk_${i+1}.wav`);
          fs.writeFileSync(chunkPath, Buffer.from(chunk.audio));
          console.log(`Chunk ${i+1}: "${chunk.text.substring(0, 40)}..." saved to ${path.basename(chunkPath)}`);
        }
      }
    }
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : String(error);
    console.error("\nError during execution:", errorMessage);
  }

  console.log('\nText-to-speech example completed successfully!');
  console.log(`Generated audio files are located in: ${outputDir}`);
}

// Execute the example
console.log('Starting TinyLM text-to-speech example...');
runTextToSpeechExample().catch(error => {
  const errorMessage = error instanceof Error ? error.message : String(error);
  console.error('Error:', errorMessage);
});
