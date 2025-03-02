/**
 * Phonemization utilities for TinyLM
 * Matches KokoroJS implementation exactly
 */

import { phonemize as espeakng } from "phonemizer";
import { split } from "./TextSplitter";

/**
 * Escapes regular expression special characters from a string
 * @param {string} string - The string to escape
 * @returns {string} Escaped string
 */
function escapeRegExp(string: string): string {
  return string.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"); // $& means the whole matched string
}

/**
 * Helper function to split numbers into phonetic equivalents
 * @param {string} match - The matched number
 * @returns {string} The phonetic equivalent
 */
function split_num(match: string): string {
  if (match.includes(".")) {
    return match;
  } else if (match.includes(":")) {
    let [h, m] = match.split(":").map(Number);
    if (h === undefined || m === undefined) return match;
    if (m === 0) {
      return `${h} o'clock`;
    } else if (m < 10) {
      return `${h} oh ${m}`;
    }
    return `${h} ${m}`;
  }

  // Handle year-like numbers (4 digits)
  const year = parseInt(match.slice(0, 4), 10);
  if (year < 1100 || year % 1000 < 10) {
    return match;
  }

  const left = match.slice(0, 2);
  const right = parseInt(match.slice(2, 4), 10);
  const suffix = match.endsWith("s") ? "s" : "";

  if (year % 1000 >= 100 && year % 1000 <= 999) {
    if (right === 0) {
      return `${left} hundred${suffix}`;
    } else if (right < 10) {
      return `${left} oh ${right}${suffix}`;
    }
  }

  return `${left} ${right}${suffix}`;
}

/**
 * Helper function to format monetary values
 * @param {string} match - The matched currency
 * @returns {string} The formatted currency
 */
function flip_money(match: string): string {
  const bill = match[0] === "$" ? "dollar" : "pound";

  if (isNaN(Number(match.slice(1)))) {
    return `${match.slice(1)} ${bill}s`;
  } else if (!match.includes(".")) {
    let suffix = match.slice(1) === "1" ? "" : "s";
    return `${match.slice(1)} ${bill}${suffix}`;
  }

  const parts = match.slice(1).split(".");
  if (parts.length < 2) return match; // Handle cases where split doesn't yield enough parts

  const [b, c] = parts;
  if (!b || !c) return match; // Extra safety check

  const d = parseInt(c.padEnd(2, "0"), 10);
  const coins = match[0] === "$" ?
    (d === 1 ? "cent" : "cents") :
    (d === 1 ? "penny" : "pence");

  return `${b} ${bill}${b === "1" ? "" : "s"} and ${d} ${coins}`;
}

/**
 * Helper function to process decimal numbers
 * @param {string} match - The matched number
 * @returns {string} The formatted number
 */
function point_num(match: string): string {
  const parts = match.split(".");
  if (parts.length < 2) return match;

  const [a, b] = parts;
  if (!a || !b) return match; // Safety check

  return `${a} point ${b.split("").join(" ")}`;
}

/**
 * Normalize text for phonemization
 * @param {string} text - The text to normalize
 * @returns {string} The normalized text
 */
function normalize_text(text: string): string {
  // Must be an exact match to KokoroJS
  return text
    // 1. Handle quotes and brackets
    .replace(/['']/g, "'")
    .replace(/«/g, "\"")
    .replace(/»/g, "\"")
    .replace(/[""]/g, '"')
    .replace(/\(/g, "«")
    .replace(/\)/g, "»")

    // 2. Replace uncommon punctuation marks
    .replace(/、/g, ", ")
    .replace(/。/g, ". ")
    .replace(/！/g, "! ")
    .replace(/，/g, ", ")
    .replace(/：/g, ": ")
    .replace(/；/g, "; ")
    .replace(/？/g, "? ")

    // 3. Whitespace normalization
    .replace(/[^\S \n]/g, " ")
    .replace(/  +/, " ")
    .replace(/(?<=\n) +(?=\n)/g, "")

    // 4. Abbreviations
    .replace(/\bD[Rr]\.(?= [A-Z])/g, "Doctor")
    .replace(/\b(?:Mr\.|MR\.(?= [A-Z]))/g, "Mister")
    .replace(/\b(?:Ms\.|MS\.(?= [A-Z]))/g, "Miss")
    .replace(/\b(?:Mrs\.|MRS\.(?= [A-Z]))/g, "Mrs")
    .replace(/\betc\.(?! [A-Z])/gi, "etc")

    // 5. Normalize casual words
    .replace(/\b(y)eah?\b/gi, "$1e'a")

    // 5. Handle numbers and currencies
    .replace(/\d*\.\d+|\b\d{4}s?\b|(?<!:)\b(?:[1-9]|1[0-2]):[0-5]\d\b(?!:)/g, split_num)
    .replace(/(?<=\d),(?=\d)/g, "")
    .replace(/[$£]\d+(?:\.\d+)?(?: hundred| thousand| (?:[bm]|tr)illion)*\b|[$£]\d+\.\d\d?\b/gi, flip_money)
    .replace(/\d*\.\d+/g, point_num)
    .replace(/(?<=\d)-(?=\d)/g, " to ")
    .replace(/(?<=\d)S/g, " S")

    // 6. Handle possessives
    .replace(/(?<=[BCDFGHJ-NP-TV-Z])'?s\b/g, "'S")
    .replace(/(?<=X')S\b/g, "s")

    // 7. Handle hyphenated words/letters
    .replace(/(?:[A-Za-z]\.){2,} [a-z]/g, (m) => m.replace(/\./g, "-"))
    .replace(/(?<=[A-Z])\.(?=[A-Z])/gi, "-")

    // 8. Strip leading and trailing whitespace
    .trim();
}

/**
 * Phonemize text using the eSpeak-NG phonemizer
 * @param {string} text - The text to phonemize
 * @param {string} language - The language code ('a' for US English, 'b' for UK English)
 * @param {boolean} norm - Whether to normalize the text
 * @returns {Promise<string>} The phonemized text
 */
export async function phonemize(text: string, language = "a", norm = true): Promise<string> {
  if (!text) return "";

  console.log(`Phonemizing text with language code '${language}'`);

  // 1. Normalize text
  if (norm) {
    text = normalize_text(text);
    console.log(`Normalized: "${text.substring(0, 50)}${text.length > 50 ? '...' : ''}"`);
  }

  // 2. Map language codes to eSpeak-NG languages - EXACTLY as in KokoroJS
  const lang = language === "a" ? "en-us" : language === "b" ? "en" : "en-us";
  console.log(`Using eSpeak-NG language: ${lang}`);

  // 3. Split into chunks, to ensure we preserve punctuation - EXACT MATCH to KokoroJS
  const PUNCTUATION = ';:,.!?¡¿—…"«»""(){}[]';
  const PUNCTUATION_PATTERN = new RegExp(`(\\s*[${escapeRegExp(PUNCTUATION)}]+\\s*)+`, "g");
  const sections = split(text, PUNCTUATION_PATTERN);

  try {
    // 4. Convert each section to phonemes
    const ps = (await Promise.all(
      sections.map(async ({ match, text }) => {
        if (match) return text; // Keep punctuation as-is
        try {
          return (await espeakng(text, lang)).join(" ");
        } catch (error) {
          console.error(`Phonemization error for text "${text}":`, error);
          return text; // Fallback to original text on error
        }
      })
    )).join("");

    // 5. Post-process phonemes - EXACTLY like KokoroJS
    let processed = ps
      .replace(/kəkˈoːɹoʊ/g, "kˈoʊkəɹoʊ")
      .replace(/kəkˈɔːɹəʊ/g, "kˈəʊkəɹəʊ")
      .replace(/ʲ/g, "j")
      .replace(/r/g, "ɹ")
      .replace(/x/g, "k")
      .replace(/ɬ/g, "l")
      .replace(/(?<=[a-zɹː])(?=hˈʌndɹɪd)/g, " ")
      .replace(/ z(?=[;:,.!?¡¿—…"«»"" ]|$)/g, "z");

    // 6. Additional post-processing for American English
    if (language === "a") {
      processed = processed.replace(/(?<=nˈaɪn)ti(?!ː)/g, "di");
    }

    console.log(`Phonemization result: "${processed.substring(0, 50)}${processed.length > 50 ? '...' : ''}"`);
    return processed.trim();

  } catch (error) {
    console.error("Phonemization failed completely:", error);
    // Return original text as fallback
    return text;
  }
}
