/**
 * TextSplitter - Splits text into sentences for better TTS quality
 */

/**
 * Escapes regular expression special characters from a string
 * @param {string} string - The string to escape
 * @returns {string} Escaped string
 */
function escapeRegExp(string: string): string {
  return string.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"); // $& means the whole matched string
}

/**
 * Split text into proper sentences for better TTS quality
 * @param {string} text - Input text to split
 * @returns {string[]} Array of sentences
 */
export function splitTextIntoSentences(text: string): string[] {
  if (!text || typeof text !== 'string') {
    return [];
  }

  // Regex for sentence boundary detection - match KokoroJS patterns exactly
  const sentenceRegex = /[.!?।॥…]+["']?\s+|[\n\r]+/g;

  // Split text on sentence boundaries
  const rawSentences = text.split(sentenceRegex);

  // Extract punctuation
  const punctuation: string[] = [];
  let match;
  while ((match = sentenceRegex.exec(text)) !== null) {
    punctuation.push(match[0]);
  }

  // Recombine sentences with their punctuation
  const sentences: string[] = [];
  for (let i = 0; i < rawSentences.length; i++) {
    let sentence = rawSentences[i]?.trim();
    if (sentence) {
      // Add back the punctuation if available
      if (i < punctuation.length) {
        sentence += punctuation[i];
      }
      sentences.push(sentence);
    }
  }

  // Handle edge cases
  const result = [];
  let currentSentence = '';
  const MAX_CHARS = 200; // Reasonable sentence length

  for (const sentence of sentences) {
    // If current sentence is getting too long, split it
    if (currentSentence && (currentSentence.length + sentence.length > MAX_CHARS)) {
      result.push(currentSentence.trim());
      currentSentence = sentence;
    } else {
      currentSentence += (currentSentence ? ' ' : '') + sentence;
    }

    // If we encounter a clear sentence boundary, push to results
    if (/[.!?][\s"']*$/.test(currentSentence)) {
      result.push(currentSentence.trim());
      currentSentence = '';
    }
  }

  // Add any remaining text
  if (currentSentence.trim()) {
    result.push(currentSentence.trim());
  }

  return result;
}

/**
 * Processes individual chunks to avoid token overflow
 * @param {string} chunk - Text chunk to check
 * @param {number} maxTokens - Maximum tokens allowed (default 500)
 * @returns {string[]} Array of safely sized chunks
 */
export function ensureSafeTokenLength(chunk: string, maxTokens: number = 500): string[] {
  if (chunk.length <= maxTokens) {
    return [chunk];
  }

  // If chunk is still too long, split it by commas, then by spaces
  if (chunk.length > maxTokens) {
    // Try splitting by commas first
    const commaSplit = chunk.split(/,\s*/);
    if (commaSplit.length > 1) {
      let results: string[] = [];
      let current = '';

      for (const part of commaSplit) {
        if ((current.length + part.length + 2) <= maxTokens) {
          current += (current ? ', ' : '') + part;
        } else {
          if (current) results.push(current);
          current = part;
        }
      }

      if (current) results.push(current);
      return results;
    }

    // If comma splitting didn't work, force split by character count
    const safeChunks: string[] = [];
    for (let i = 0; i < chunk.length; i += maxTokens) {
      // Try to find a space near the boundary
      let endPos = Math.min(i + maxTokens, chunk.length);
      if (endPos < chunk.length) {
        const spacePos = chunk.lastIndexOf(' ', endPos);
        if (spacePos > i) {
          endPos = spacePos;
        }
      }
      safeChunks.push(chunk.substring(i, endPos).trim());
    }
    return safeChunks;
  }

  return [chunk];
}

/**
 * Helper function to split a string on a regex, but keep the delimiters.
 * This matches KokoroJS implementation exactly
 * @param {string} text - The text to split
 * @param {RegExp} regex - The regex to split on
 * @returns {{match: boolean; text: string}[]} The split string
 */
export function split(text: string, regex: RegExp): Array<{match: boolean; text: string}> {
  const result: Array<{match: boolean; text: string}> = [];
  let prev = 0;

  for (const match of text.matchAll(regex)) {
    const fullMatch = match[0];
    if (prev < match.index!) {
      result.push({ match: false, text: text.slice(prev, match.index) });
    }
    if (fullMatch.length > 0) {
      result.push({ match: true, text: fullMatch });
    }
    prev = match.index! + fullMatch.length;
  }

  if (prev < text.length) {
    result.push({ match: false, text: text.slice(prev) });
  }

  return result;
}
