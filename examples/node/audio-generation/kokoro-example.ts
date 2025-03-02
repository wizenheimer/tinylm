/**
 * Text-to-Speech Example with TinyLM
 *
 * This example demonstrates the speech generation capabilities:
 * - Hardware capability detection
 * - TTS model loading with progress tracking
 * - Speech generation with different voices
 * - Speed adjustments
 * - Saving audio to files
 * - Multiple languages support
 */

import { TinyLM, ProgressUpdate, FileInfo, OverallProgress } from '../../../src/index';
import * as fs from 'fs';
import * as path from 'path';
import { fileURLToPath } from 'url';

// Get current directory (ES module compatible)
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Format bytes to human-readable size
function formatBytes(bytes: number | undefined): string {
  if (bytes === 0 || !bytes) return '0 B';
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(1024));
  return `${(bytes / Math.pow(1024, i)).toFixed(2)} ${sizes[i]}`;
}

// Format seconds to human-readable time
function formatTime(seconds: number | null): string {
  if (!seconds || seconds === 0) return '';
  if (seconds < 60) return `${Math.ceil(seconds)}s`;
  if (seconds < 3600) {
    const minutes = Math.floor(seconds / 60);
    const secs = Math.ceil(seconds % 60);
    return `${minutes}m ${secs}s`;
  }
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  return `${hours}h ${minutes}m`;
}

// Format overall progress information nicely
function formatProgress(progress: ProgressUpdate): string {
  const { type, status, percentComplete, message, files, overall } = progress;

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

  // Color based on type
  let color = '';
  let resetColor = '';
  if (typeof process !== 'undefined' && process.stdout &&
    // TypeScript-safe check for hasColors method
    typeof (process.stdout as any).hasColors === 'function' &&
    (process.stdout as any).hasColors()) {
    // Terminal colors
    switch (type) {
      case 'system': color = '\x1b[36m'; break; // Cyan
      case 'tts_model': color = '\x1b[32m'; break; // Green
      case 'speech': color = '\x1b[35m'; break; // Magenta
      default: color = ''; break;
    }
    resetColor = '\x1b[0m';
  }

  // Format output lines
  let output = `${color}[${status}]${resetColor} ${type ? `(${type})` : ''}`;
  if (progressBar) output += ` ${progressBar}`;
  if (message) output += ` ${message}`;

  // Add overall stats if available
  const overallProgress = overall as OverallProgress | undefined;
  if (overallProgress && type === 'tts_model' && status === 'loading') {
    output += `\n  Total: ${overallProgress.formattedLoaded}/${overallProgress.formattedTotal}`;
    if (overallProgress.formattedSpeed) {
      output += ` at ${overallProgress.formattedSpeed}`;
    }
    if (overallProgress.formattedRemaining) {
      output += ` - ETA: ${overallProgress.formattedRemaining}`;
    }
  }

  // Add file-specific progress if available
  if (Array.isArray(files) && files.length > 0 && type === 'tts_model') {
    // Show active files first
    const activeFiles = files.filter(f => f.status !== 'done' && f.status !== 'error');
    if (activeFiles.length > 0) {
      output += '\n  Active downloads:';
      activeFiles.forEach((file: FileInfo) => {
        output += `\n    ${file.name}: ${file.percentComplete}% (${formatBytes(file.bytesLoaded)}/${formatBytes(file.bytesTotal)})`;
        if (file.speed > 0) {
          output += ` at ${formatBytes(file.speed)}/s`;
        }
        if (file.timeRemaining) {
          output += ` - ETA: ${formatTime(file.timeRemaining)}`;
        }
      });
    }

    // Show recently completed files (last 2)
    const doneFiles = files.filter(f => f.status === 'done').slice(-2);
    if (doneFiles.length > 0) {
      output += '\n  Recently completed:';
      doneFiles.forEach((file: FileInfo) => {
        output += `\n    ${file.name}: Complete (${formatBytes(file.bytesTotal)})`;
      });
    }
  }

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

// Main text-to-speech example
async function runTextToSpeechExample(): Promise<void> {
  console.log('=== TinyLM Text-to-Speech Example ===');

  // Create an output directory
  const outputDir = ensureOutputDirExists();

  // Create a new TinyLM instance with custom progress tracking
  const tiny = new TinyLM({
    progressCallback: (progress: ProgressUpdate) => {
      try {
        console.log(formatProgress(progress));
      } catch (error) {
        // Fallback to simple logging
        console.log(`[${progress.status}] ${progress.message || ''}`);
        console.error('Error formatting progress:', error);
      }
    },
    progressThrottleTime: 100, // Update frequently to show progress
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
      ttsModels: ['onnx-community/Kokoro-82M-v1.0-ONNX'], // The TTS model from the implementation
    });

    // Example 1: Basic speech generation
    console.log("\n=== Example 1: Basic Speech Generation ===");

    const text = "Hello world! This is an example of text-to-speech with TinyLM.";
    console.log(`\nGenerating speech for: "${text}"`);

    const outputPath = path.join(outputDir, 'basic_speech.wav');

    const result = await tiny.audio.speech.create({
      model: 'onnx-community/Kokoro-82M-v1.0-ONNX',
      input: text,
      voice: 'af', // Default voice
      response_format: 'wav'
    });

    // Save the audio buffer to a file
    fs.writeFileSync(outputPath, Buffer.from(result.audio));
    console.log(`\nSpeech saved to: ${outputPath}`);
    console.log(`Generation time: ${result._tinylm?.time_ms}ms`);

    // Example 2: Using different voices
    console.log("\n=== Example 2: Different Voices ===");

    // Create a function to generate speech with different voices
    async function generateWithVoice(voice: string): Promise<void> {
      console.log(`\nGenerating speech with voice: ${voice}`);
      const text = `This is an example of the ${voice} voice.`;

      const result = await tiny.audio.speech.create({
        model: 'onnx-community/Kokoro-82M-v1.0-ONNX',
        input: text,
        voice,
        response_format: 'wav'
      });

      const outputPath = path.join(outputDir, `${voice}_example.wav`);
      fs.writeFileSync(outputPath, Buffer.from(result.audio));
      console.log(`Speech saved to: ${outputPath}`);
    }

    // Generate examples with different voices
    await generateWithVoice('af_bella');   // American female
    await generateWithVoice('am_adam');    // American male
    await generateWithVoice('bf_emma');    // British female

    // Example 3: Speed adjustment
    console.log("\n=== Example 3: Speed Adjustment ===");

    const speedText = "This is a demonstration of different speech speeds.";

    async function generateWithSpeed(speed: number): Promise<void> {
      console.log(`\nGenerating speech with speed: ${speed}`);

      const result = await tiny.audio.speech.create({
        model: 'onnx-community/Kokoro-82M-v1.0-ONNX',
        input: speedText,
        voice: 'af_bella',
        speed,
        response_format: 'wav'
      });

      const outputPath = path.join(outputDir, `speed_${speed.toString().replace('.', '_')}.wav`);
      fs.writeFileSync(outputPath, Buffer.from(result.audio));
      console.log(`Speech saved to: ${outputPath}`);
    }

    // Generate examples with different speeds
    await generateWithSpeed(0.8); // Slower
    await generateWithSpeed(1.0); // Normal
    await generateWithSpeed(1.2); // Faster

    // Example 4: Multi-language support
    console.log("\n=== Example 4: Multi-language Support ===");

    async function generateInLanguage(text: string, voice: string, description: string): Promise<void> {
      console.log(`\nGenerating speech in ${description}`);

      const result = await tiny.audio.speech.create({
        model: 'onnx-community/Kokoro-82M-v1.0-ONNX',
        input: text,
        voice,
        response_format: 'wav'
      });

      const outputPath = path.join(outputDir, `${voice}_language_example.wav`);
      fs.writeFileSync(outputPath, Buffer.from(result.audio));
      console.log(`Speech saved to: ${outputPath}`);
    }

    // Generate examples in different languages
    await generateInLanguage("Hello, this is English text with an American accent.", "af_bella", "American English");
    await generateInLanguage("Hello, this is English text with a British accent.", "bf_emma", "British English");
    await generateInLanguage("Hola, este es un texto en español.", "ef_dora", "Spanish");
    await generateInLanguage("नमस्ते, यह हिंदी में एक उदाहरण है।", "hf_alpha", "Hindi");

    // Example 5: Advanced use - Generate a paragraph with natural pauses
    console.log("\n=== Example 5: Advanced Use - Paragraph with Natural Pauses ===");

    const paragraph = "Welcome to the world of speech synthesis. Artificial voices have come a long way. They now sound much more natural and expressive. This technology enables many accessibility features. It's also used in virtual assistants and automated systems.";

    // Split into sentences and add pauses
    const sentences = paragraph.split('.');
    const sentencesWithPauses = sentences.map(s => s.trim()).filter(s => s.length > 0);

    console.log(`\nGenerating paragraph speech with ${sentencesWithPauses.length} sentences`);

    // Generate speech for the full paragraph
    const paragraphResult = await tiny.audio.speech.create({
      model: 'onnx-community/Kokoro-82M-v1.0-ONNX',
      input: paragraph,
      voice: 'af_bella',
      response_format: 'wav'
    });

    const paragraphOutputPath = path.join(outputDir, 'paragraph_speech.wav');
    fs.writeFileSync(paragraphOutputPath, Buffer.from(paragraphResult.audio));
    console.log(`Full paragraph speech saved to: ${paragraphOutputPath}`);

    // Example 6: Offloading the model
    console.log("\n=== Example 6: Model Management ===");

    // Get the list of loaded models
    console.log("\nCurrently loaded TTS models:", tiny.models.listTTS());

    // Offload the model
    console.log("\nOffloading TTS model...");
    const offloadResult = await tiny.models.offloadTTS({
      model: 'onnx-community/Kokoro-82M-v1.0-ONNX'
    });

    console.log("Model offloaded:", offloadResult);
    console.log("TTS models still loaded:", tiny.models.listTTS());

    // Re-load the model
    console.log("\nRe-loading TTS model...");
    await tiny.models.loadTTS({
      model: 'onnx-community/Kokoro-82M-v1.0-ONNX'
    });

    console.log("TTS models after reloading:", tiny.models.listTTS());

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
