//! Text chunking utilities for streaming TTS.
//!
//! Splits input text into chunks optimized for streaming audio generation.
//! The first chunk is split aggressively at the earliest clause or sentence
//! boundary to minimize time-to-first-audio.

/// Split text into chunks optimized for streaming TTS.
///
/// The first chunk is split at the earliest clause/sentence boundary within
/// `first_max` characters to minimize time-to-first-audio. Remaining text
/// uses sentence-level chunking with `rest_max` as the limit.
pub fn chunk_text_streaming(text: &str, first_max: usize, rest_max: usize) -> Vec<String> {
    let text = text.trim();
    if text.is_empty() {
        return Vec::new();
    }

    // Short text → single chunk
    if text.len() <= first_max {
        return vec![ensure_punctuation(text)];
    }

    // Find the earliest clause or sentence boundary within first_max chars.
    // We search within a char-safe prefix to avoid splitting multi-byte chars.
    let search_end = text
        .char_indices()
        .take_while(|(i, _)| *i < first_max)
        .last()
        .map(|(i, c)| i + c.len_utf8())
        .unwrap_or(text.len().min(first_max));
    let search_region = &text[..search_end];

    let clause_delimiters = [", ", "; ", ": ", " — ", " - "];
    let sentence_endings = [". ", "! ", "? "];

    let mut earliest_split: Option<usize> = None;

    // Check clause delimiters: split after the punctuation, before the space
    for delim in &clause_delimiters {
        if let Some(pos) = search_region.find(delim) {
            let split_at = pos + delim.len() - 1; // after punct, before space
            earliest_split = Some(match earliest_split {
                Some(prev) if prev <= split_at => prev,
                _ => split_at,
            });
        }
    }

    // Check sentence endings: split after the punctuation + space
    for delim in &sentence_endings {
        if let Some(pos) = search_region.find(delim) {
            let split_at = pos + delim.len(); // after the space
            earliest_split = Some(match earliest_split {
                Some(prev) if prev <= split_at => prev,
                _ => split_at,
            });
        }
    }

    // Also check sentence ending at end of search region (no trailing space)
    for &ending in &['.', '!', '?'] {
        if search_region.ends_with(ending) {
            let split_at = search_region.len();
            earliest_split = Some(match earliest_split {
                Some(prev) if prev <= split_at => prev,
                _ => split_at,
            });
        }
    }

    match earliest_split {
        Some(pos) if pos > 0 => {
            let first = text[..pos].trim();
            let rest = text[pos..].trim();
            let mut chunks = vec![ensure_punctuation(first)];
            if !rest.is_empty() {
                chunks.extend(chunk_text(rest, rest_max));
            }
            chunks
        }
        _ => {
            // No boundary found — split at last word boundary before first_max
            let words: Vec<&str> = text.split_whitespace().collect();
            let mut len = 0;
            let mut split_words = 0;
            for (i, w) in words.iter().enumerate() {
                let next_len = len + w.len() + if i > 0 { 1 } else { 0 };
                if next_len > first_max {
                    break;
                }
                len = next_len;
                split_words = i + 1;
            }
            if split_words == 0 {
                split_words = 1;
            }

            let first_text: String = words[..split_words].join(" ");
            let rest_text: String = words[split_words..].join(" ");
            let mut chunks = vec![ensure_punctuation(&first_text)];
            if !rest_text.is_empty() {
                chunks.extend(chunk_text(&rest_text, rest_max));
            }
            chunks
        }
    }
}

/// Split text into chunks at sentence boundaries, with a maximum length.
pub fn chunk_text(text: &str, max_len: usize) -> Vec<String> {
    let text = text.trim();
    if text.is_empty() {
        return Vec::new();
    }

    if text.len() <= max_len {
        return vec![ensure_punctuation(text)];
    }

    // Split on sentence boundaries: `. ` `! ` `? ` or end-of-string after `.!?`
    let mut chunks = Vec::new();
    let mut remaining = text;

    while !remaining.is_empty() {
        if remaining.len() <= max_len {
            chunks.push(ensure_punctuation(remaining));
            break;
        }

        // Look for the last sentence boundary within max_len
        let search_end = remaining
            .char_indices()
            .take_while(|(i, _)| *i < max_len)
            .last()
            .map(|(i, c)| i + c.len_utf8())
            .unwrap_or(max_len.min(remaining.len()));
        let search_region = &remaining[..search_end];

        let mut best_split = None;
        for &ending in &[". ", "! ", "? "] {
            if let Some(pos) = search_region.rfind(ending) {
                let split_at = pos + ending.len();
                best_split = Some(match best_split {
                    Some(prev) if prev > split_at => prev,
                    _ => split_at,
                });
            }
        }

        match best_split {
            Some(pos) => {
                let chunk = remaining[..pos].trim();
                if !chunk.is_empty() {
                    chunks.push(ensure_punctuation(chunk));
                }
                remaining = remaining[pos..].trim();
            }
            None => {
                // No sentence boundary — split at last word boundary
                let words: Vec<&str> = search_region.split_whitespace().collect();
                if words.len() <= 1 {
                    // Single very long word; take it as-is
                    chunks.push(ensure_punctuation(search_region));
                    remaining = remaining[search_end..].trim();
                } else {
                    let split_text: String = words[..words.len() - 1].join(" ");
                    let split_len = split_text.len();
                    chunks.push(ensure_punctuation(&split_text));
                    remaining = remaining[split_len..].trim();
                }
            }
        }
    }

    chunks
}

/// Ensure the text ends with punctuation. Appends a comma if it doesn't.
///
/// This helps TTS models produce natural-sounding prosody at chunk boundaries.
fn ensure_punctuation(text: &str) -> String {
    let text = text.trim();
    if text.is_empty() {
        return String::new();
    }

    let last_char = text.chars().last().unwrap();
    if matches!(last_char, '.' | '!' | '?' | ',' | ';' | ':') {
        text.to_string()
    } else {
        format!("{},", text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_short_text_single_chunk() {
        let chunks = chunk_text_streaming("Hello.", 100, 400);
        assert_eq!(chunks, vec!["Hello."]);
    }

    #[test]
    fn test_empty_text() {
        let chunks = chunk_text_streaming("", 100, 400);
        assert!(chunks.is_empty());
    }

    #[test]
    fn test_aggressive_first_chunk_comma() {
        // Use first_max=50 to force splitting on this 64-char text
        let chunks = chunk_text_streaming(
            "Hello, how are you doing today? I hope everything is going well.",
            50,
            400,
        );
        assert_eq!(chunks[0], "Hello,");
        assert!(chunks.len() >= 2);
    }

    #[test]
    fn test_aggressive_first_chunk_sentence() {
        let chunks = chunk_text_streaming(
            "Hello world. This is a longer text that should be split into chunks for streaming delivery.",
            50,
            400,
        );
        // Should split at ". " — earliest boundary
        assert_eq!(chunks[0], "Hello world.");
        assert!(chunks.len() >= 2);
    }

    #[test]
    fn test_long_text_first_chunk_aggressive() {
        // Text > 100 chars with an early comma
        let chunks = chunk_text_streaming(
            "Hello, this is a much longer text that exceeds one hundred characters. It should split at the very first comma which appears early.",
            100,
            400,
        );
        assert_eq!(chunks[0], "Hello,");
        assert!(chunks.len() >= 2);
    }

    #[test]
    fn test_ensure_punctuation() {
        assert_eq!(ensure_punctuation("Hello"), "Hello,");
        assert_eq!(ensure_punctuation("Hello."), "Hello.");
        assert_eq!(ensure_punctuation("Hello!"), "Hello!");
        assert_eq!(ensure_punctuation("Hello,"), "Hello,");
    }

    #[test]
    fn test_no_delimiter_word_boundary_split() {
        let chunks = chunk_text_streaming(
            "This text has no early delimiter and goes on for quite a while without any comma or period",
            30,
            400,
        );
        assert!(chunks[0].len() <= 35); // first_max + punctuation
        assert!(chunks.len() >= 2);
    }

    #[test]
    fn test_chunk_text_sentence_split() {
        let chunks = chunk_text("First sentence. Second sentence. Third sentence.", 30);
        assert!(chunks.len() >= 2);
        assert!(chunks[0].contains("First"));
    }
}
