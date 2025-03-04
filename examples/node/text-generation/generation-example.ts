/**
 * Text Generation Example with TinyLM
 *
 * This example demonstrates the text generation capabilities:
 * - Hardware capability detection
 * - Model loading with detailed per-file progress tracking
 * - Regular completions
 * - True streaming responses
 * - Interrupting generation
 * - Advanced parameter configuration
 * - Multi-turn conversations
 */

import { TinyLM } from '../../../src/TinyLM';
import { ProgressTracker } from '../../../src/ProgressTracker';
import { WebGPUChecker } from '../../../src/WebGPUChecker';
import { ModelType } from '../../../src/types';
import type { ProgressUpdate, CompletionChunk, FileInfo, OverallProgress, CompletionResult } from '../../../src/types';

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
      case 'model': color = '\x1b[32m'; break; // Green
      case 'generation': color = '\x1b[35m'; break; // Magenta
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
  if (overallProgress && type === 'model' && status === 'loading') {
    output += `\n  Total: ${overallProgress.formattedLoaded}/${overallProgress.formattedTotal}`;
    if (overallProgress.formattedSpeed) {
      output += ` at ${overallProgress.formattedSpeed}`;
    }
    if (overallProgress.formattedRemaining) {
      output += ` - ETA: ${overallProgress.formattedRemaining}`;
    }
  }

  // Add file-specific progress if available
  if (Array.isArray(files) && files.length > 0 && type === 'model') {
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

// Type guard to check if an object is an AsyncGenerator
function isAsyncGenerator(obj: any): obj is AsyncGenerator<CompletionChunk> {
  return obj && typeof obj[Symbol.asyncIterator] === 'function';
}

// Main text generation example
async function runTextGenerationExample(): Promise<void> {
  console.log('=== TinyLM Text Generation Example ===');

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
    progressCallback: (progress: ProgressUpdate) => {
      try {
        console.log(formatProgress(progress));
      } catch (error) {
        // Fallback to simple logging
        console.log(`[${progress.status}] ${progress.message || ''}`);
      }
    },
    progressThrottleTime: 100
  });

  try {
    // Initialize TinyLM
    console.log('\nInitializing TinyLM...');
    await tiny.init({
      models: ['HuggingFaceTB/SmolLM2-135M-Instruct'], // Specify the model during initialization
      lazyLoad: false // Load the model immediately
    });

    // Example 1: Basic non-streaming completion
    console.log("\n=== Example 1: Basic Completion ===");
    const response = await tiny.chat.completions.create({
      model: 'HuggingFaceTB/SmolLM2-135M-Instruct',
      messages: [
        { role: "system", content: "You are a helpful AI assistant." },
        { role: "user", content: "What is artificial intelligence?" }
      ],
      temperature: 0.7,
      max_tokens: 100
    });

    if (!isAsyncGenerator(response)) {
      const content = response.choices[0]?.message?.content;
      if (content) {
        console.log("\nResponse:", content);
      }
    }

    // Example 2: Streaming completion
    console.log("\n=== Example 2: Streaming Completion ===");
    const stream = await tiny.chat.completions.create({
      model: 'HuggingFaceTB/SmolLM2-135M-Instruct',
      messages: [
        { role: "system", content: "You are a helpful AI assistant." },
        { role: "user", content: "Explain the concept of machine learning in simple terms." }
      ],
      temperature: 0.7,
      max_tokens: 150,
      stream: true
    });

    // Example 3: Multi-turn conversation
    console.log("\n=== Example 3: Multi-turn Conversation ===");

    // Start a conversation history
    const conversation = [
      { role: "system", content: "You are a knowledgeable assistant specializing in programming." },
      { role: "user", content: "What is a pure function in programming?" }
    ];

    console.log("\nUser: What is a pure function in programming?");

    // First response
    const turn1 = await tiny.chat.completions.create({
      model: 'HuggingFaceTB/SmolLM2-135M-Instruct',
      messages: conversation,
      temperature: 0.3,
      max_tokens: 150
    });

    if (!isAsyncGenerator(turn1)) {
      const assistantResponse = turn1.choices[0]?.message.content;
      console.log(`\nAssistant: ${assistantResponse}`);

      // Add the assistant's response to the conversation history
      conversation.push({
        role: "assistant",
        content: assistantResponse as string
      });

      // Add the next user message
      conversation.push({
        role: "user",
        content: "Can you give me a simple example of a pure function in JavaScript?"
      });

      console.log("\nUser: Can you give me a simple example of a pure function in JavaScript?");

      // Second response
      const turn2 = await tiny.chat.completions.create({
        model: 'HuggingFaceTB/SmolLM2-135M-Instruct',
        messages: conversation,
        temperature: 0.3,
        max_tokens: 150
      });

      if (!isAsyncGenerator(turn2)) {
        console.log(`\nAssistant: ${turn2.choices[0]?.message.content}`);
      }
    }

    // Example 4: Interrupting generation
    console.log("\n=== Example 4: Interrupting Generation ===");

    const longMessages = [
      { role: "system", content: "You are a detailed technical writer." },
      { role: "user", content: "Write a detailed guide on machine learning algorithms." }
    ];

    console.log("\nStarting long generation (will interrupt after 3 seconds)...");

    // Start generation
    const longGeneration = tiny.chat.completions.create({
      model: 'HuggingFaceTB/SmolLM2-135M-Instruct',
      messages: longMessages,
      temperature: 0.7,
      max_tokens: 500,
      stream: true,
    });

    // Set timeout to interrupt generation
    setTimeout(() => {
      console.log("\n\nInterrupting generation...");
      tiny.models.interrupt();
    }, 3000);

    console.log("\nResponse:");
    let interruptedResponse = '';

    try {
      const streamResult = await longGeneration;
      if (isAsyncGenerator(streamResult)) {
        for await (const chunk of streamResult) {
          const content = chunk.choices[0]?.delta?.content || '';
          interruptedResponse += content;

          if (content) {
            process.stdout.write(content);
          }
        }
      }
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      console.log("\n\nGeneration was interrupted:", errorMessage);
    }

    // Reset generation state
    tiny.models.reset();
    console.log("\nGeneration state reset");

    // Example 5: Continuation generation (efficient token generation)
    console.log("\n=== Example 5: Continuation Generation ===");

    const initialPrompt = "Write a short story about a robot.";
    console.log("\nGenerating the first part of a story...");

    const part1 = await tiny.chat.completions.create({
      model: 'HuggingFaceTB/SmolLM2-135M-Instruct',
      messages: [{ role: "user", content: initialPrompt }],
      temperature: 0.8,
      max_tokens: 100 // Limit tokens to show continuation
    });

    if (!isAsyncGenerator(part1)) {
      const part1Content = part1.choices[0]?.message.content;
      console.log("\nPart 1:");
      console.log(part1Content);

      console.log("\nContinuing the story (using cached key values)...");

      // Continuation should be efficient using cached key values
      const part2 = await tiny.chat.completions.create({
        model: 'HuggingFaceTB/SmolLM2-135M-Instruct',
        messages: [
          { role: "user", content: initialPrompt },
          { role: "assistant", content: part1Content as string },
          { role: "user", content: "Continue the story:" }
        ],
        temperature: 0.8,
        max_tokens: 100
      });

      if (!isAsyncGenerator(part2)) {
        console.log("\nPart 2:");
        console.log(part2.choices[0]?.message.content);
      }
    }

    // Example 6: Advanced generation parameters
    console.log("\n=== Example 6: Advanced Parameters ===");

    const userPrompt = "List 3 emerging technologies in AI.";
    console.log(`\nPrompt: "${userPrompt}"`);

    // Sample with different temperatures
    console.log("\nWith temperature = 0.2 (more focused):");
    const lowTempResponse = await tiny.chat.completions.create({
      model: 'HuggingFaceTB/SmolLM2-135M-Instruct',
      messages: [{ role: "user", content: userPrompt }],
      temperature: 0.2,
      max_tokens: 150
    });

    if (!isAsyncGenerator(lowTempResponse)) {
      console.log(lowTempResponse.choices[0]?.message.content);
    }

    console.log("\nWith temperature = 1.0 (more creative):");
    const highTempResponse = await tiny.chat.completions.create({
      model: 'HuggingFaceTB/SmolLM2-135M-Instruct',
      messages: [{ role: "user", content: userPrompt }],
      temperature: 1.0,
      max_tokens: 150
    });

    if (!isAsyncGenerator(highTempResponse)) {
      console.log(highTempResponse.choices[0]?.message.content);
    }

    // Example with top_p sampling
    console.log("\nWith top_p = 0.5 (nucleus sampling):");
    const topPResponse = await tiny.chat.completions.create({
      model: 'HuggingFaceTB/SmolLM2-135M-Instruct',
      messages: [{ role: "user", content: userPrompt }],
      temperature: 0.7,
      top_p: 0.5,
      max_tokens: 150
    });

    if (!isAsyncGenerator(topPResponse)) {
      console.log(topPResponse.choices[0]?.message.content);
    }

    // Example 7: Offloading a model
    console.log("\n=== Example 7: Model Offloading ===");

    console.log("\nOffloading model...");
    const offloadResult = await tiny.models.offload({
      model: 'HuggingFaceTB/SmolLM2-135M-Instruct'
    });

    console.log("Model offloaded:", offloadResult);
    console.log("Models still loaded:", tiny.models.list());

  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : String(error);
    console.error("\nError during execution:", errorMessage);
  }

  console.log('\nText generation example completed');
}

// Execute the example
console.log('Starting TinyLM text generation example...');
runTextGenerationExample().catch(error => {
  const errorMessage = error instanceof Error ? error.message : String(error);
  console.error('Error:', errorMessage);
});
