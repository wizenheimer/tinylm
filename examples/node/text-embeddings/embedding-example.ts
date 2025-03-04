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

import { TinyLM } from '../../../src/TinyLM';
import { WebGPUChecker } from '../../../src/WebGPUChecker';
import { ProgressTracker } from '../../../src/ProgressTracker';
import { ModelType } from '../../../src/types';
import type { ProgressUpdate } from '../../../src/types';

// Format bytes to human-readable size
function formatBytes(bytes: number | undefined): string {
  if (bytes === 0 || !bytes) return '0 B';
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(1024));
  return `${(bytes / Math.pow(1024, i)).toFixed(2)} ${sizes[i]}`;
}

// Helper function to format progress updates
function formatProgress(progress: ProgressUpdate): string {
  const parts = [`[${progress.status}]`];
  if (progress.type) parts.push(`(${progress.type})`);
  if (typeof progress.progress === 'number') {
    const percent = Math.round(progress.progress * 100);
    parts.push(`[${'#'.repeat(percent / 5)}${'-'.repeat(20 - percent / 5)}] ${percent}%`);
  }
  if (progress.message) parts.push(progress.message);
  return parts.join(' ');
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
  try {
    console.log('=== TinyLM Text Embeddings Example ===\n');

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

    // Initialize TinyLM with the embedding model
    await tiny.init();

    // Load the embedding model
    console.log('\nLoading embedding model...');
    await tiny.models.load({
      model: 'Xenova/all-MiniLM-L6-v2',
      type: ModelType.Embedding
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

    // Example 3: Semantic search
    console.log("\n=== Example 3: Semantic Search ===");
    const documents = [
      "Artificial intelligence is a branch of computer science.",
      "Machine learning is a subset of artificial intelligence.",
      "Neural networks are used in deep learning.",
      "GPUs accelerate machine learning computations.",
      "Natural language processing helps computers understand text.",
      "The global climate is changing due to human activities.",
      "Renewable energy sources include solar, wind, and hydro power."
    ];

    console.log(`\nIndexing ${documents.length} documents for semantic search...`);
    const documentEmbed = await tiny.embeddings.create({
      model: 'Xenova/all-MiniLM-L6-v2',
      input: documents
    });

    const documentEmbeddings = documentEmbed.data.map(d => d.embedding) as number[][];

    // Search function
    async function semanticSearch(query: string, topK: number = 3) {
      const queryEmbed = await tiny.embeddings.create({
        model: 'Xenova/all-MiniLM-L6-v2',
        input: query
      });

      const queryEmbedding = queryEmbed.data[0]?.embedding as number[];
      const similarities = documentEmbeddings.map((docEmbedding, index) => ({
        index,
        similarity: cosineSimilarity(queryEmbedding, docEmbedding),
        text: documents[index]
      }));

      similarities.sort((a, b) => b.similarity - a.similarity);
      return similarities.slice(0, topK);
    }

    // Example queries
    const queries = [
      "What is machine learning?",
      "Tell me about renewable energy",
      "How do neural networks work?"
    ];

    for (const query of queries) {
      console.log(`\nSearch query: "${query}"`);
      const results = await semanticSearch(query);
      console.log("Top 3 matching documents:");
      results.forEach((result, i) => {
        console.log(`${i + 1}. (${(result.similarity * 100).toFixed(1)}%) ${result.text}`);
      });
    }

    // Cleanup
    console.log('\nCleaning up...');
    await tiny.models.offload({
      model: 'Xenova/all-MiniLM-L6-v2',
      type: ModelType.Embedding
    });
    console.log('Model unloaded successfully');

  } catch (error) {
    console.error('Error:', error);
  }
}

// Run the example
runTextEmbeddingsExample().catch(console.error);
