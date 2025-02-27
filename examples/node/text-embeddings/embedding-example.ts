/**
 * Text Embeddings Example with TinyLM
 *
 * This example demonstrates the text embeddings capabilities:
 * - Hardware capability detection
 * - Detailed model loading progress tracking
 * - Single and batch embedding generation
 * - Different encoding formats (float vs base64)
 * - Semantic search implementation
 * - Document clustering
 * - Nearest neighbors finding
 */

import { TinyLM, ProgressUpdate, FileInfo, OverallProgress } from '../../../src/index';

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
      case 'embedding': color = '\x1b[33m'; break; // Yellow
      case 'embedding_model': color = '\x1b[34m'; break; // Blue
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
  if (overallProgress && (type === 'model' || type === 'embedding_model') && status === 'loading') {
    output += `\n  Total: ${overallProgress.formattedLoaded}/${overallProgress.formattedTotal}`;
    if (overallProgress.formattedSpeed) {
      output += ` at ${overallProgress.formattedSpeed}`;
    }
    if (overallProgress.formattedRemaining) {
      output += ` - ETA: ${overallProgress.formattedRemaining}`;
    }
  }

  // Add file-specific progress if available
  if (Array.isArray(files) && files.length > 0 && (type === 'model' || type === 'embedding_model')) {
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

// Calculate cosine similarity between two vectors
function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length !== b.length) {
    throw new Error("Vectors must have the same dimensions");
  }

  let dotProduct = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < a.length; i++) {
    dotProduct += a[i]! * b[i]!;
    normA += a[i]! * a[i]!;
    normB += b[i]! * b[i]!;
  }

  return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}

// Main text embeddings example
async function runTextEmbeddingsExample(): Promise<void> {
  console.log('=== TinyLM Text Embeddings Example ===');

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
      embeddingModels: ['Xenova/all-MiniLM-L6-v2'], // Specify embedding model
      lazyLoad: true, // Don't load model yet
    });

    // Example 1: Single text embedding
    console.log("\n=== Example 1: Single Text Embedding ===");

    const sampleText = "The quick brown fox jumps over the lazy dog";
    console.log(`\nGenerating embedding for text: "${sampleText}"`);

    const singleResult = await tiny.embeddings.create({
      model: 'Xenova/all-MiniLM-L6-v2',
      input: sampleText
    });

    const embedding = singleResult.data[0]?.embedding as number[];

    console.log(`\nEmbedding generated:`);
    console.log(`- Dimensions: ${embedding.length}`);
    console.log(`- Token usage: ${singleResult.usage.prompt_tokens} tokens`);
    console.log(`- First 5 values: [${embedding.slice(0, 5).map(v => v.toFixed(6)).join(', ')}]`);
    console.log(`- Last 5 values: [${embedding.slice(-5).map(v => v.toFixed(6)).join(', ')}]`);

    // Example 2: Batch embedding for multiple texts
    console.log("\n=== Example 2: Batch Embedding ===");

    const texts = [
      "Artificial intelligence is transforming technology",
      "Machine learning models need large amounts of data",
      "Neural networks are inspired by the human brain",
      "The weather looks beautiful today"
    ];

    console.log(`\nGenerating embeddings for ${texts.length} texts in a single batch...`);

    const batchStartTime = Date.now();
    const batchResult = await tiny.embeddings.create({
      model: 'Xenova/all-MiniLM-L6-v2',
      input: texts
    });
    const batchTimeTaken = Date.now() - batchStartTime;

    console.log(`\nBatch embeddings generated in ${batchTimeTaken}ms (${batchTimeTaken / texts.length}ms per text):`);
    console.log(`- Number of embeddings: ${batchResult.data.length}`);
    console.log(`- Embedding dimensions: ${(batchResult.data[0]?.embedding as number[]).length}`);
    console.log(`- Total token usage: ${batchResult.usage.prompt_tokens} tokens`);

    // Print first few elements of each embedding
    console.log('\nFirst few elements of each embedding:');
    batchResult.data.forEach((item, idx) => {
      const embedding = item.embedding as number[];
      console.log(`Text ${idx + 1}: [${embedding.slice(0, 3).map(v => v.toFixed(4)).join(', ')}...]`);
    });

    // Example 3: Different encoding formats
    console.log("\n=== Example 3: Encoding Formats ===");

    const formatText = "This is a test of different embedding formats";

    console.log(`\nGenerating embeddings in different formats for: "${formatText}"`);

    // Default float embeddings
    const floatResult = await tiny.embeddings.create({
      model: 'Xenova/all-MiniLM-L6-v2',
      input: formatText,
      encoding_format: 'float' // Default
    });

    // Base64 encoded embeddings (compact format for storage/transmission)
    const base64Result = await tiny.embeddings.create({
      model: 'Xenova/all-MiniLM-L6-v2',
      input: formatText,
      encoding_format: 'base64'
    });

    const floatEmbedding = floatResult.data[0]?.embedding as number[];
    const base64Embedding = base64Result.data[0]?.embedding as string;

    console.log("\nFloat format:");
    console.log(`- Type: ${typeof floatEmbedding[0]}`);
    console.log(`- Dimensions: ${floatEmbedding.length}`);
    console.log(`- Sample: [${floatEmbedding.slice(0, 3).map(v => v.toFixed(6)).join(', ')}...]`);
    console.log(`- Size as JSON: ${JSON.stringify(floatEmbedding).length} bytes`);

    console.log("\nBase64 format:");
    console.log(`- Type: ${typeof base64Embedding}`);
    console.log(`- Length: ${base64Embedding.length} chars`);
    console.log(`- Sample: ${base64Embedding.substring(0, 50)}...`);
    console.log(`- Size: ${base64Embedding.length} bytes`);

    // Compare storage efficiency
    const compressionRatio = base64Embedding.length / JSON.stringify(floatEmbedding).length * 100;
    console.log(`\nBase64 size is ${compressionRatio.toFixed(1)}% of float JSON format`);

    // Example 4: Semantic search
    console.log("\n=== Example 4: Semantic Search ===");

    // Collection of documents to search
    const documents = [
      "Artificial intelligence is a branch of computer science.",
      "Machine learning is a subset of artificial intelligence.",
      "Neural networks are used in deep learning.",
      "GPUs accelerate machine learning computations.",
      "Natural language processing helps computers understand text.",
      "Computer vision focuses on image and video analysis.",
      "Transformer architecture revolutionized NLP tasks.",
      "The Turing test evaluates a machine's ability to exhibit human-like intelligence.",
      "The global climate is changing due to human activities.",
      "Renewable energy sources include solar, wind, and hydro power."
    ];

    console.log(`\nIndexing ${documents.length} documents for semantic search...`);

    // Generate embeddings for all documents (our "index")
    const documentEmbed = await tiny.embeddings.create({
      model: 'Xenova/all-MiniLM-L6-v2',
      input: documents
    });

    const documentEmbeddings = documentEmbed.data.map(d => d.embedding) as number[][];

    // Print sample embeddings
    console.log('\nSample document embeddings (first 3 documents, first 3 dimensions):');
    documentEmbeddings.slice(0, 3).forEach((embedding, idx) => {
      console.log(`Doc ${idx + 1}: [${embedding.slice(0, 3).map(v => v.toFixed(4)).join(', ')}...]`);
    });

    // Search function
    async function semanticSearch(query: string, topK: number = 3) {
      // Generate embedding for the query
      const queryEmbed = await tiny.embeddings.create({
        model: 'Xenova/all-MiniLM-L6-v2',
        input: query
      });

      const queryEmbedding = queryEmbed.data[0]?.embedding as number[];

      // Calculate similarity to all documents
      const similarities = documentEmbeddings.map((docEmbedding, index) => {
        const similarity = cosineSimilarity(queryEmbedding, docEmbedding);
        return { index, similarity, text: documents[index] };
      });

      // Sort by similarity (descending)
      similarities.sort((a, b) => b.similarity - a.similarity);

      // Return top K results
      return similarities.slice(0, topK);
    }

    // Example queries
    const queries = [
      "What is machine learning?",
      "Tell me about neural networks",
      "How is climate change affecting the planet?"
    ];

    // Run semantic search for each query
    for (const query of queries) {
      console.log(`\nSearch query: "${query}"`);

      const results = await semanticSearch(query);

      console.log("Top 3 matching documents:");
      results.forEach((result, i) => {
        console.log(`${i+1}. (${(result.similarity * 100).toFixed(1)}%) ${result.text}`);
      });
    }

    // Example 5: Finding similar texts using embeddings
    console.log("\n=== Example 5: Finding Similar Texts ===");

    const sentences = [
      "The cat sat on the mat",
      "The dog sat on the rug",
      "The feline rested on the carpet",
      "Quantum physics explores the nature of reality",
      "Artificial intelligence systems can learn from data",
      "Deep learning models require substantial computing power"
    ];

    console.log(`\nGenerating embeddings for ${sentences.length} sentences to find similarities...`);

    const sentenceEmbeddings = await tiny.embeddings.create({
      model: 'Xenova/all-MiniLM-L6-v2',
      input: sentences
    });

    const embeddings = sentenceEmbeddings.data.map(d => d.embedding as number[]);

    // Calculate pairwise similarities
    console.log("\nPairwise similarities:");

    for (let i = 0; i < sentences.length; i++) {
      for (let j = i + 1; j < sentences.length; j++) {
        const similarity = cosineSimilarity(embeddings[i]!, embeddings[j]!);
        console.log(`"${sentences[i]}" ↔ "${sentences[j]}": ${(similarity * 100).toFixed(1)}%`);
      }
    }

    // Find the most similar pair
    let maxSimilarity = 0;
    let maxPair = [0, 0];

    for (let i = 0; i < sentences.length; i++) {
      for (let j = i + 1; j < sentences.length; j++) {
        const similarity = cosineSimilarity(embeddings[i]!, embeddings[j]!);
        if (similarity > maxSimilarity) {
          maxSimilarity = similarity;
          maxPair = [i, j];
        }
      }
    }

    console.log(`\nMost similar pair (${(maxSimilarity * 100).toFixed(1)}%):`);
    console.log(`1. "${sentences[maxPair[0] ?? 0]}"`);
    console.log(`2. "${sentences[maxPair[1] ?? 0]}"`);

    // Example 6: Offloading the embedding model
    console.log("\n=== Example 6: Offloading Embedding Model ===");

    console.log("\nOffloading embedding model...");

    // Note: We would add an offloadModel method to the EmbeddingsModule
    // This assumes it exists in the implementation
    // const offloadSuccess = await tiny.embeddings.offloadModel('Xenova/all-MiniLM-L6-v2');
    // console.log(`Model offloaded: ${offloadSuccess}`);

  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : String(error);
    console.error("\nError during execution:", errorMessage);
  }

  console.log('\nText embeddings example completed');
}

// Execute the example
console.log('Starting TinyLM text embeddings example...');
runTextEmbeddingsExample().catch(error => {
  const errorMessage = error instanceof Error ? error.message : String(error);
  console.error('Error:', errorMessage);
});
