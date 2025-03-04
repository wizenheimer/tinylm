/**
 * Nomic AI Embeddings Example with TinyLM
 *
 * This example demonstrates using Nomic AI's powerful embedding models:
 * - Loading Nomic's high-performance text embedding model
 * - Using WebGPU acceleration when available
 * - Text similarity comparison
 * - Zero-shot classification
 * - Cross-language similarities
 * - Document retrieval and ranking
 */

import { TinyLM } from '../../../src/TinyLM';
import { WebGPUChecker } from '../../../src/WebGPUChecker';
import { ProgressTracker } from '../../../src/ProgressTracker';
import { ModelType } from '../../../src/types';
import type { ProgressUpdate } from '../../../src/types';
import { FileInfo, OverallProgress } from '../../../src/index';

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

// Format progress information
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

// Main Nomic embeddings example
async function runNomicEmbeddingsExample(): Promise<void> {
  console.log('=== TinyLM with Nomic AI Embeddings Example ===');

  // Check hardware capabilities
  console.log('\nChecking hardware capabilities...');
  const webgpuChecker = new WebGPUChecker();
  const capabilities = await webgpuChecker.check();
  console.log(`WebGPU available: ${capabilities.isWebGPUSupported}`);
  console.log(`FP16 supported: ${capabilities.fp16Supported}`);
  console.log(`Backend: ${capabilities.adapterInfo?.name || 'unknown'}\n`);

  // Create a new TinyLM instance with progress tracking
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

  try {
    // Initialize TinyLM
    console.log('\nInitializing TinyLM...');
    await tiny.init();

    // Load the Nomic embedding model
    console.log('\nLoading Nomic embedding model...');
    await tiny.models.load({
      model: 'nomic-ai/nomic-embed-text-v1.5',
      type: ModelType.Embedding
    });

    const NOMIC_MODEL = 'nomic-ai/nomic-embed-text-v1.5';

    // Example 1: Basic Nomic Text Embeddings
    console.log("\n=== Example 1: Basic Nomic Text Embeddings ===");

    const text = "Nomic AI develops powerful embedding models for language understanding.";
    console.log(`\nGenerating Nomic embedding for: "${text}"`);

    const result = await tiny.embeddings.create({
      model: NOMIC_MODEL,
      input: text
    });

    const embedding = result.data[0]?.embedding as number[];

    console.log(`\nNomic embedding generated:`);
    console.log(`- Dimensions: ${embedding.length}`);
    console.log(`- Token usage: ${result.usage.prompt_tokens} tokens`);
    console.log(`- First 5 values: [${embedding.slice(0, 5).map(v => v.toFixed(6)).join(', ')}]`);

    // Example 2: Semantic Similarity with Nomic
    console.log("\n=== Example 2: Semantic Similarity with Nomic Embeddings ===");

    const sentencePairs = [
      // Similar pairs
      ["Neural networks process information in layers.", "Deep learning models transform inputs through multiple processing stages."],
      ["The company's stock price increased after the earnings report.", "Following financial results, the firm's shares traded higher."],
      // Dissimilar pairs
      ["Quantum mechanics describes the behavior of subatomic particles.", "The chef prepared a delicious Mediterranean dinner."],
      ["Global climate change affects biodiversity in ecosystems.", "Solving algebraic equations requires understanding variables."]
    ];

    console.log("\nComparing semantic similarity between sentence pairs using Nomic embeddings...");

    for (const [sentence1, sentence2] of sentencePairs) {
      // Get embeddings for both sentences
      const embedResults = await tiny.embeddings.create({
        model: NOMIC_MODEL,
        input: [sentence1!, sentence2!]
      });

      const embedding1 = embedResults.data[0]?.embedding as number[];
      const embedding2 = embedResults.data[1]?.embedding as number[];

      // Calculate similarity
      const similarity = cosineSimilarity(embedding1, embedding2);

      console.log(`\nSentence 1: "${sentence1}"`);
      console.log(`Sentence 2: "${sentence2}"`);
      console.log(`Similarity: ${(similarity * 100).toFixed(2)}%`);
    }

    // Example 3: Multilingual Embeddings
    console.log("\n=== Example 3: Multilingual Capabilities ===");

    const multilingualTexts = [
      { language: "English", text: "Artificial intelligence is changing the world." },
      { language: "Spanish", text: "La inteligencia artificial está cambiando el mundo." },
      { language: "French", text: "L'intelligence artificielle change le monde." },
      { language: "German", text: "Künstliche Intelligenz verändert die Welt." },
      { language: "Chinese", text: "人工智能正在改变世界。" },
    ];

    console.log("\nComparing cross-language semantic similarity with Nomic embeddings...");

    // Generate embeddings for all texts
    const multilingualEmbeddings = await tiny.embeddings.create({
      model: NOMIC_MODEL,
      input: multilingualTexts.map(item => item.text)
    });

    const embeddings = multilingualEmbeddings.data.map(d => d.embedding) as number[][];

    // Compare similarity between English and each other language
    const englishEmbedding = embeddings[0]!;

    console.log("\nSimilarity between English and other languages (same meaning):");
    for (let i = 1; i < multilingualTexts.length; i++) {
      const similarity = cosineSimilarity(englishEmbedding, embeddings[i]!);
      console.log(`- English → ${multilingualTexts[i]!.language}: ${(similarity * 100).toFixed(2)}%`);
    }

    // Example 4: Zero-Shot Text Classification
    console.log("\n=== Example 4: Zero-Shot Classification ===");

    // Define classes and their descriptions
    const classes = [
      { label: "Technology", description: "Content about computers, software, AI, and technological advancements" },
      { label: "Health", description: "Content about medicine, wellness, diseases, and healthcare systems" },
      { label: "Finance", description: "Content about money, investing, markets, banking, and economics" },
      { label: "Entertainment", description: "Content about movies, music, celebrities, and leisure activities" }
    ];

    // Documents to classify
    const documents = [
      "Apple announced its new M3 chip with significant performance improvements for MacBooks.",
      "Researchers discovered a promising new treatment for Type 2 diabetes in clinical trials.",
      "The Federal Reserve decided to keep interest rates unchanged after their meeting yesterday.",
      "The latest Marvel movie broke box office records during its opening weekend."
    ];

    console.log("\nPerforming zero-shot classification with Nomic embeddings...");

    // Generate embeddings for class descriptions
    const classEmbeddings = await tiny.embeddings.create({
      model: NOMIC_MODEL,
      input: classes.map(c => c.description)
    });

    // Get class vectors
    const classVectors = classEmbeddings.data.map(d => d.embedding) as number[][];

    // Classify each document
    for (const document of documents) {
      console.log(`\nDocument: "${document}"`);

      // Generate document embedding
      const docEmbedding = await tiny.embeddings.create({
        model: NOMIC_MODEL,
        input: document
      });

      const docVector = docEmbedding.data[0]?.embedding as number[];

      // Calculate similarity to each class
      const predictions = classes.map((cls, i) => {
        const similarity = cosineSimilarity(docVector, classVectors[i]!);
        return { label: cls.label, score: similarity };
      });

      // Sort by score (descending)
      predictions.sort((a, b) => b.score - a.score);

      // Display predictions
      console.log("Classification results:");
      predictions.forEach((pred, i) => {
        console.log(`  ${i + 1}. ${pred.label}: ${(pred.score * 100).toFixed(2)}%`);
      });

      console.log(`Classified as: ${predictions[0]?.label}`);
    }

    // Example 5: Document Retrieval
    console.log("\n=== Example 5: Document Retrieval ===");

    // Create a small corpus of documents
    const corpus = [
      { id: "doc1", content: "Nomic AI develops state-of-the-art embedding models for various AI applications." },
      { id: "doc2", content: "Large language models have transformed natural language processing in recent years." },
      { id: "doc3", content: "Neural networks require significant computational resources for training." },
      { id: "doc4", content: "Vector embeddings allow for efficient semantic search in large document collections." },
      { id: "doc5", content: "Transformer architecture introduced attention mechanisms that revolutionized NLP." },
      { id: "doc6", content: "Data quality is crucial for developing robust machine learning models." },
      { id: "doc7", content: "Embedding models map text to high-dimensional vector spaces where semantic relationships are preserved." },
      { id: "doc8", content: "Fine-tuning pre-trained models can adapt them to specific domains or tasks." }
    ];

    console.log(`\nCreating vector index for ${corpus.length} documents...`);

    // Generate embeddings for all documents
    const corpusEmbeddings = await tiny.embeddings.create({
      model: NOMIC_MODEL,
      input: corpus.map(doc => doc.content)
    });

    const documentVectors = corpusEmbeddings.data.map(d => d.embedding) as number[][];

    // Show sample embeddings
    console.log('\nSample document embeddings (first 3 documents, first 5 dimensions):');
    documentVectors.slice(0, 3).forEach((vec, idx) => {
      console.log(`Doc ${idx + 1}: [${vec.slice(0, 5).map(v => v.toFixed(4)).join(', ')}...]`);
    });

    // Simple retrieval function
    async function retrieveDocuments(query: string, topK: number = 3) {
      // Generate query embedding
      const queryEmbedResult = await tiny.embeddings.create({
        model: NOMIC_MODEL,
        input: query
      });

      const queryVector = queryEmbedResult.data[0]?.embedding as number[];

      // Show query embedding
      console.log(`\nQuery embedding (first 5 dimensions):`);
      console.log(`[${queryVector.slice(0, 5).map(v => v.toFixed(4)).join(', ')}...]`);

      // Calculate similarities
      const similarities = documentVectors.map((docVec, i) => ({
        id: corpus[i]!.id,
        content: corpus[i]!.content,
        similarity: cosineSimilarity(queryVector, docVec)
      }));

      // Sort by similarity
      similarities.sort((a, b) => b.similarity - a.similarity);

      // Return top K
      return similarities.slice(0, topK);
    }

    // Example queries
    const queries = [
      "How do embedding models work?",
      "What are the computational requirements for AI?",
      "Tell me about fine-tuning models"
    ];

    // Run queries
    for (const query of queries) {
      console.log(`\nQuery: "${query}"`);

      const results = await retrieveDocuments(query);

      console.log("Top 3 relevant documents:");
      results.forEach((doc, i) => {
        console.log(`  ${i + 1}. [${doc.id}] (${(doc.similarity * 100).toFixed(2)}%): ${doc.content}`);
      });
    }

    // Example 6: Comparing with other embedding models
    console.log("\n=== Example 6: Model Comparison ===");

    // Load another embedding model for comparison
    const COMPARISON_MODEL = 'Xenova/all-MiniLM-L6-v2';
    await tiny.models.load({
      model: COMPARISON_MODEL,
      type: ModelType.Embedding
    });

    console.log(`\nComparing Nomic embeddings with ${COMPARISON_MODEL}...`);

    const comparisonTexts = [
      "The model effectively captures semantic meaning in language.",
      "The large language model responds to complex queries quickly.",
      "Embedding models represent text as high-dimensional vectors."
    ];

    // Generate embeddings with both models
    console.log("\nGenerating embeddings with both models for the same texts...");

    // Nomic embeddings
    const nomicResults = await tiny.embeddings.create({
      model: NOMIC_MODEL,
      input: comparisonTexts
    });

    // Comparison model embeddings
    const otherResults = await tiny.embeddings.create({
      model: COMPARISON_MODEL,
      input: comparisonTexts
    });

    // Check if dimensionality is different
    const nomicDims = (nomicResults.data[0]?.embedding as number[]).length;
    const otherDims = (otherResults.data[0]?.embedding as number[]).length;

    console.log(`\nDimensionality comparison:`);
    console.log(`- Nomic (${NOMIC_MODEL}): ${nomicDims} dimensions`);
    console.log(`- Other (${COMPARISON_MODEL}): ${otherDims} dimensions`);

    // Compare within-model similarities
    console.log("\nWithin-model similarity patterns:");

    // Function to display similarity matrix
    function displaySimilarityMatrix(embeddings: number[][], modelName: string) {
      console.log(`\n${modelName} similarities:`);

      // Header row
      console.log(`${"".padEnd(10)}| ${comparisonTexts.map((_, i) => `Text ${i + 1}`.padEnd(10)).join(" | ")}`);
      console.log("-".repeat(10 + (comparisonTexts.length * 13)));

      // Rows
      for (let i = 0; i < embeddings.length; i++) {
        let row = `Text ${i + 1}`.padEnd(10) + "| ";

        for (let j = 0; j < embeddings.length; j++) {
          const similarity = cosineSimilarity(embeddings[i]!, embeddings[j]!);
          row += (j === i) ? "1.000".padEnd(10) : similarity.toFixed(3).padEnd(10);
          row += " | ";
        }

        console.log(row);
      }
    }

    // Display similarity matrices
    displaySimilarityMatrix(
      nomicResults.data.map(d => d.embedding) as number[][],
      "Nomic"
    );

    displaySimilarityMatrix(
      otherResults.data.map(d => d.embedding) as number[][],
      COMPARISON_MODEL
    );

    // Cleanup comparison model
    await tiny.models.offload({
      model: COMPARISON_MODEL,
      type: ModelType.Embedding
    });

    // Cleanup
    console.log('\nCleaning up...');
    await tiny.models.offload({
      model: NOMIC_MODEL,
      type: ModelType.Embedding
    });
    console.log('Model unloaded successfully');

  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : String(error);
    console.error("\nError during execution:", errorMessage);
  }

  console.log('\nNomic embeddings example completed');
}

// Execute the example
console.log('Starting TinyLM with Nomic AI embeddings example...');
runNomicEmbeddingsExample().catch(error => {
  const errorMessage = error instanceof Error ? error.message : String(error);
  console.error('Error:', errorMessage);
});
