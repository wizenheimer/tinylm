/**
 * Text-to-Speech Example with TinyLM
 *
 * This demonstrates speech generation with TinyLM, including:
 * - Basic speech generation
 * - Streaming for better handling of long texts
 * - Comparing streaming vs non-streaming approaches
 */

import { TinyLM } from '../../../src/TinyLM';
import { WebGPUChecker } from '../../../src/WebGPUChecker';
import { ProgressTracker } from '../../../src/ProgressTracker';
import { ModelType, SpeechResult } from '../../../src/types';
import * as fs from 'fs';
import * as path from 'path';
import { fileURLToPath } from 'url';
import { writeFile } from 'fs/promises';

// Get current directory (ES module compatible)
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Helper function to format progress updates
function formatProgress(progress: any): string {
  const parts = [`[${progress.status}]`];
  if (progress.type) parts.push(`(${progress.type})`);
  if (typeof progress.progress === 'number') {
    const percent = Math.round(progress.progress * 100);
    parts.push(`[${'#'.repeat(percent / 5)}${'-'.repeat(20 - percent / 5)}] ${percent}%`);
  }
  if (progress.message) parts.push(progress.message);
  return parts.join(' ');
}

// Main example function
async function runTextToSpeechExample() {
  try {
    console.log('=== TinyLM Text-to-Speech Example ===\n');

    // Check hardware capabilities
    console.log('Checking hardware capabilities...');
    const webgpuChecker = new WebGPUChecker();
    const capabilities = await webgpuChecker.check();
    console.log(`WebGPU available: ${capabilities.isWebGPUSupported}`);
    console.log(`FP16 supported: ${capabilities.fp16Supported}`);
    console.log(`Backend: ${capabilities.adapterInfo?.name || 'unknown'}\n`);

    // Create a new TinyLM instance with custom progress tracking
    console.log('Initializing TinyLM...');
    const tiny = new TinyLM({
      progressTracker: new ProgressTracker((progress) => {
        try {
          console.log(formatProgress(progress));
        } catch (error) {
          // Fallback to simple logging
          console.log(`[${progress.status}] ${progress.message || ''}`);
        }
      }),
      webgpuChecker
    });

    // Initialize TinyLM with the Kokoro model
    await tiny.init({
      ttsModels: ['onnx-community/Kokoro-82M-v1.0-ONNX'],
      lazyLoad: false // Load the model immediately
    });

    // Generate speech
    console.log('\nGenerating speech...');
    const text = "Hello, this is a test of text to speech generation.";
    const result = await tiny.audio.speech.create({
      input: text,
      voice: 'af', // Default voice
      response_format: 'wav',
      stream: false // Ensure we get a non-streaming result
    }) as SpeechResult; // Type assertion since we know it's not streaming

    // Save the audio to a file
    const outputPath = 'output.wav';
    await writeFile(outputPath, Buffer.from(result.audio));
    console.log(`\nAudio saved to ${outputPath}`);

    // Example 1: Basic non-streaming speech generation
    console.log('\nGenerating basic speech...');
    const shortText = 'Hello! This is a test of the TinyLM text-to-speech system.';
    const basicResult = await tiny.audio.speech.create({
      input: shortText,
      voice: 'af',
      response_format: 'mp3'
    });

    // Save basic result
    if ('audio' in basicResult) {
      const basicOutputPath = path.join(__dirname, 'output', 'basic_speech.mp3');
      await fs.promises.mkdir(path.dirname(basicOutputPath), { recursive: true });
      await fs.promises.writeFile(basicOutputPath, Buffer.from(basicResult.audio));
      console.log(`Basic speech saved to: ${basicOutputPath}`);
    }

    // Example 2: Streaming speech generation for longer text
    console.log('\nGenerating streaming speech...');
    const longText = `
      The quick brown fox jumps over the lazy dog.
      This is a longer piece of text that demonstrates streaming speech generation.
      Each sentence will be processed separately and returned as a chunk.
      This allows for real-time playback and progress tracking.
    `.trim();

    const streamResult = await tiny.audio.speech.create({
      input: longText,
      voice: 'af',
      response_format: 'mp3',
      stream: true
    });

    // Save streaming chunks
    if ('chunks' in streamResult) {
      for (const chunk of streamResult.chunks) {
        const chunkPath = path.join(__dirname, 'output', `stream_chunk_${streamResult.chunks.indexOf(chunk) + 1}.mp3`);
        await fs.promises.writeFile(chunkPath, Buffer.from(chunk.audio));
        console.log(`Chunk ${streamResult.chunks.indexOf(chunk) + 1} saved to: ${chunkPath}`);
        console.log(`Text: "${chunk.text}"`);
      }
    }

    // Cleanup
    console.log('\nCleaning up...');
    await tiny.models.offload({
      model: 'onnx-community/Kokoro-82M-v1.0-ONNX',
      type: ModelType.Audio
    });
    console.log('Model unloaded successfully');

  } catch (error) {
    console.error('Error:', error);
  }
}

// Run the example
runTextToSpeechExample().catch(console.error);
