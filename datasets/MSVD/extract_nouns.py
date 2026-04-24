"""
Extract visual atomic units from video captions using GlobalAI API.
Usage: python extract_nouns.py --input val_preprocessed.txt --output annotations/nouns/val
"""
import os
import sys
import json
import time
import argparse
import requests
import re
import numpy as np
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from pathlib import Path

# Add project root to path for importing Tokenizer_M
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent / "project"
sys.path.insert(0, str(PROJECT_ROOT))

API_URL = "https://globalai.vip/v1/chat/completions"
MODEL = "gpt-4.1-nano"
KEYS_FILE = ".keys"

# Global tokenizer (lazy loaded)
_TOKENIZER = None


def get_tokenizer():
    """Lazy load the extended tokenizer."""
    global _TOKENIZER
    if _TOKENIZER is None:
        from load_tokenizers import Tokenizer_M
        tokenizer_path = PROJECT_ROOT / "models" / "tokenizer_m" / "tokenizer"
        print(f"Loading tokenizer from: {tokenizer_path}")
        _TOKENIZER = Tokenizer_M.from_pretrained(str(tokenizer_path))
        print(f"Tokenizer vocab size: {_TOKENIZER.vocab_size}")
    return _TOKENIZER


def create_noun_vector(nouns: list, tokenizer) -> np.ndarray:
    """Create a multi-hot vector for a list of nouns.
    
    Args:
        nouns: List of noun strings
        tokenizer: The tokenizer instance
    
    Returns:
        numpy array of shape [vocab_size] with 1s at noun token positions
    """
    vocab_size = tokenizer.vocab_size
    vector = np.zeros(vocab_size, dtype=np.float32)
    
    for noun in nouns:
        # Tokenize without special tokens, use first token
        tokens = tokenizer.encode(noun, add_special_tokens=False)
        if tokens and tokens[0] < vocab_size:
            vector[tokens[0]] = 1.0
    
    return vector


def decode_noun_vector(vector: np.ndarray, tokenizer=None) -> dict:
    """Decode a multi-hot noun vector back to token IDs and words.
    
    Args:
        vector: numpy array of shape [vocab_size] with 1s at noun positions
        tokenizer: The tokenizer instance (if None, will lazy load)
    
    Returns:
        dict with keys:
            - 'ids': list of token IDs where vector is non-zero
            - 'tokens': list of token strings
            - 'count': number of active tokens
    
    Usage:
        >>> vector = np.load("annotations/noun_vectors/val/video_id.npy")
        >>> result = decode_noun_vector(vector)
        >>> print(result['tokens'])  # ['man', 'car', 'dog', ...]
        >>> print(result['ids'])     # [786, 1203, 2456, ...]
    """
    if tokenizer is None:
        tokenizer = get_tokenizer()
    
    # Get non-zero indices
    ids = np.nonzero(vector)[0].tolist()
    
    # Convert to tokens
    tokens = tokenizer.convert_ids_to_tokens(ids)
    
    return {
        'ids': ids,
        'tokens': tokens,
        'count': len(ids),
    }


def decode_noun_vector_file(npy_path: str) -> dict:
    """Convenience function to decode a noun vector from a .npy file.
    
    Args:
        npy_path: Path to the .npy file
    
    Returns:
        dict with 'ids', 'tokens', and 'count'
    
    Usage:
        >>> result = decode_noun_vector_file("annotations/noun_vectors/val/video_id.npy")
        >>> print(result['tokens'])
    """
    vector = np.load(npy_path)
    return decode_noun_vector(vector)

SYSTEM_PROMPT = """You are extracting visual atomic units from multiple captions of the same video.

For each caption, extract only nouns, plural nouns, or compound nouns that represent a single visible object/entity.

## Extraction Rules (STRICT):

1. KEEP ONLY:
   - Concrete object nouns (man, dog, bike, tree)
   - Plural nouns for visible entities (dogs, people)
   - Compound nouns that represent a single visual object (ice cream, toy car, traffic light)

2. REMOVE ALL modifiers:
   - adjectives (small, black, sunny, tall)
   - adverbs (quickly, happily)
   - quantity words (one, two, several, many)
   - determiners (a, the, this, that)
   - abstract/mental verbs (trying, thinking)
   - scene-level non-object descriptors (sunny, rainy, beautiful)

3. IMPORTANT: Even if the caption contains actions, you MUST still extract visible entity nouns (e.g., person, man, woman, people, cat, dog). Never return an empty list if there is at least one visible entity noun in the caption.

4. SURFACE FORM must be preserved exactly:
   - Do NOT convert plural to singular
   - Do NOT change gerund forms
   - Do NOT lemmatize
   - Do NOT split or merge phrases except when removing modifiers

5. VALID UNIT CRITERIA:
   - Must correspond to a *single* visually identifiable entity/object (NOT an action)
   - Multi-word phrase allowed ONLY if it forms a single atomic object/entity
     (e.g., "ice cream", "toy car", "traffic light")

6. FORBIDDEN (DO NOT OUTPUT):

- Any verb, gerund, or action phrase
  (e.g., "pouring", "running", "playing drums", "pouring liquid into plastic bag", "marinating chicken")
- Any verb-object phrase of any length

6. Output must follow the required JSON-like format:


## OUTPUT FORMAT:

{
  "video_objs": [all unique visual units across all captions],
  "sentence_objs": {
     "sentence_id1": [units for caption 1],
     "sentence_id2": [units for caption 2],
     ...
  }
}

- sentence-level lists must preserve original order of appearance in the caption.
- video_objs must be the union of all sentence-level units (duplicates removed).
- All tokens must be strings.

## DEDUPLICATION (IMPORTANT):

- In each sentence-level list, remove duplicate units if the same unit appears multiple times in that caption. Preserve the first occurrence order.
- In video_objs, remove duplicates across captions. Preserve the first occurrence order.


## NOW PROCESS THE FOLLOWING CAPTIONS:

{caption_block}


Return ONLY the JSON object exactly in the specified structure."""


OBJ_PHRASES_PROMPT = """You are given:

1) A list of video-level object units (nouns / compound nouns): {video_objs}
2) A set of captions from the same video (multiple sentences):

{caption_block}

Task:

For EACH object unit in video_objs, find all mentions in the captions and extract the minimal *object-only related phrase* from that caption.

Definition of "minimal object-only related phrase":

- It must be the smallest contiguous phrase that still clearly refers to the object in that caption.
- It MAY include attributes/modifiers and MAY include an action involving the object.
- CRITICAL: the phrase must contain the target object and MUST NOT contain any other concrete object/entity noun.
  If the caption contains multiple objects, keep only the part about the target object and DROP the other object(s).
  Also drop background/location prepositional phrases (e.g., "on the field", "in the room") unless the target object itself is the location noun.

Examples (required behavior):

- obj="men":
  - input: "two men are hugging a lion" -> output phrase: "two men are hugging"
  - input: "two men are playing with a lion" -> output phrase: "two men are playing"
- obj="lion":
  - input: "a lion hugs two men" -> output phrase: "a lion hugs"
  - input: "a lion is playing with a couple" -> output phrase: "a lion is playing"

Keep surface form as it appears in the caption (do not lemmatize).

Rules:

0. Phrase matching mode:

- Mode A (STRICT LITERAL): a phrase MUST literally contain the object string (case-insensitive).
  Example: obj="man" -> keep "a man in black"; drop "the guy".
- Mode B (SEMANTIC): phrases may use synonyms or coreference if they clearly refer to the object.
  Example: obj="man" -> allow "the guy" if it clearly refers to the man.

Current mode: {phrase_mode}

1. A phrase can be:
   - noun phrase with modifiers/attributes
   - short action phrase that includes the object (e.g., "man playing drums")
2. Deduplicate phrases per object, preserving first occurrence order.
3. Limit: return at most {max_phrases_per_obj} phrases per object.
4. If an object does not appear in any caption, return an empty list for that object.

Output format (JSON only):

{
  "video_objs": {
    "obj1": ["phrase1", "phrase2"],
    "obj2": [],
    ...
  }
}

Return ONLY the JSON."""


# NOTE: Currently we only need video-level annotations (video_objs).
# Sentence-level extraction (sentence_objs) is intentionally disabled below.


def load_keys(path: str):
    """Load API keys from .keys file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Keys file not found: {path}")
    keys = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            line = line.split("#", 1)[0].strip()
            if line:
                keys.append(line)
    return keys


def load_captions(input_path: str):
    """Load and group captions by video_id."""
    videos = defaultdict(list)
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(" ", 2)
            if len(parts) < 3:
                continue
            video_id, sent_id, caption = parts[0], parts[1], parts[2]
            videos[video_id].append((sent_id, caption))
    return videos


def build_caption_block(captions):
    """Build caption block for the prompt."""
    lines = []
    for sent_id, caption in captions:
        lines.append(f"{sent_id}: {caption}")
    return "\n".join(lines)


def call_api(key: str, caption_block: str, max_retries: int = 3):
    """Call GlobalAI API to extract visual atomic units."""
    prompt = SYSTEM_PROMPT.replace("{caption_block}", caption_block)
    
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": "Please extract visual atomic units from the captions above and return the JSON."}
        ],
        "temperature": 0.2,
        "max_tokens": 2048,
    }
    
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    
    for attempt in range(max_retries):
        try:
            resp = requests.post(API_URL, json=payload, headers=headers, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            return content, None
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                return None, str(e)
    return None, "Max retries exceeded"


def call_obj_phrases_api(
    key: str,
    video_objs: list,
    caption_block: str,
    phrase_mode: str = "A",
    max_phrases_per_obj: int = 8,
    max_retries: int = 3,
):
    """Call GlobalAI API to map each video_obj to minimal related phrases across captions."""
    prompt = OBJ_PHRASES_PROMPT.replace("{video_objs}", json.dumps(video_objs, ensure_ascii=False))
    prompt = prompt.replace("{caption_block}", caption_block)
    prompt = prompt.replace("{max_phrases_per_obj}", str(max_phrases_per_obj))
    prompt = prompt.replace("{phrase_mode}", phrase_mode)

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": "Return the JSON mapping now."},
        ],
        "temperature": 0.2,
        "max_tokens": 2048,
    }

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    for attempt in range(max_retries):
        try:
            resp = requests.post(API_URL, json=payload, headers=headers, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            return content, None
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                return None, str(e)
    return None, "Max retries exceeded"


def parse_obj_phrases_response(response: str):
    """Parse video_objs mapping (obj -> phrases) JSON from API response."""
    result, err = parse_json_response(response)
    if err:
        return None, err
    if not isinstance(result, dict) or "video_objs" not in result or not isinstance(result["video_objs"], dict):
        return None, "Invalid video_objs mapping format"
    return result["video_objs"], None


def dedup_preserve_order(items):
    if not isinstance(items, list):
        return []
    seen = set()
    out = []
    for x in items:
        if isinstance(x, str) and x not in seen:
            seen.add(x)
            out.append(x)
    return out


def trim_phrase_remove_other_objs(obj: str, phrase: str, all_objs: list) -> str:
    """Mode A: enforce single-object by removing any other object keys present in the phrase.

    Heuristic: if the phrase contains another object key (from all_objs) besides obj,
    truncate the phrase before the earliest such occurrence.
    """
    if not isinstance(obj, str) or not isinstance(phrase, str):
        return ""
    if not isinstance(all_objs, list):
        all_objs = []

    p = " ".join(phrase.strip().split())
    if not p:
        return ""

    obj_l = obj.strip().lower()
    p_l = p.lower()
    if not obj_l or obj_l not in p_l:
        return ""

    # Find earliest occurrence of any OTHER object key.
    cut_pos = None
    for o in all_objs:
        if not isinstance(o, str):
            continue
        o_l = o.strip().lower()
        if not o_l or o_l == obj_l:
            continue
        idx = p_l.find(o_l)
        if idx == -1:
            continue
        # ensure it looks like a token/phrase boundary (avoid accidental substring hits)
        left_ok = (idx == 0) or (p_l[idx - 1] in " \t\n\r\f\v,.;:!?\"'()[]{}")
        right_idx = idx + len(o_l)
        right_ok = (right_idx >= len(p_l)) or (p_l[right_idx] in " \t\n\r\f\v,.;:!?\"'()[]{}")
        if not (left_ok and right_ok):
            continue
        if cut_pos is None or idx < cut_pos:
            cut_pos = idx

    if cut_pos is not None and cut_pos > 0:
        p = p[:cut_pos].strip()

    # Remove trailing determiners / dangling prepositions that may remain
    bad_tail = (
        " a", " an", " the",
        " of", " to", " in", " on", " at", " by", " for", " from", " with", " into", " onto", " over", " under",
    )
    while True:
        lower = p.lower()
        changed = False
        for bt in bad_tail:
            if lower.endswith(bt):
                p = p[: -len(bt)].strip()
                changed = True
                break
        if not changed:
            break

    if obj_l not in p.lower():
        return ""
    return p


def fallback_extract_obj_patient_phrase(obj: str, phrase: str, all_objs: list) -> str:
    """Fallback when obj is typically a patient (direct object), e.g. "playing drums".

    Try to return a short substring that still contains obj but excludes other object keys.
    Strategy:
    - Find the obj span.
    - Expand left to include up to two tokens (often a verb / aux + verb).
    - Then ensure no other object keys are present.
    """
    if not isinstance(obj, str) or not isinstance(phrase, str):
        return ""
    obj_l = obj.strip().lower()
    if not obj_l:
        return ""

    p = " ".join(phrase.strip().split())
    if not p:
        return ""
    p_l = p.lower()
    idx = p_l.find(obj_l)
    if idx == -1:
        return ""

    tokens = p.split(" ")
    # map char idx -> token index (simple scan)
    char = 0
    obj_tok_i = None
    for i, tok in enumerate(tokens):
        start = char
        end = char + len(tok)
        if start <= idx < end:
            obj_tok_i = i
            break
        char = end + 1
    if obj_tok_i is None:
        return ""

    start_i = max(0, obj_tok_i - 2)
    candidate = " ".join(tokens[start_i: obj_tok_i + 1]).strip()

    # remove trailing determiners/prepositions if any
    bad_tail = {"a", "an", "the", "of", "to", "in", "on", "at", "by", "for", "from", "with", "into", "onto", "over", "under"}
    while candidate and candidate.lower().split()[-1] in bad_tail:
        candidate = " ".join(candidate.split()[:-1]).strip()

    if obj_l not in candidate.lower():
        return ""

    # ensure it doesn't include other object keys
    cand_l = candidate.lower()
    for o in all_objs if isinstance(all_objs, list) else []:
        if not isinstance(o, str):
            continue
        o_l = o.strip().lower()
        if not o_l or o_l == obj_l:
            continue
        if o_l in cand_l:
            return ""

    return candidate


def trim_phrase_single_object(obj: str, phrase: str) -> str:
    """Trim a phrase so it focuses on the target object only (heuristic)."""
    if not isinstance(obj, str) or not isinstance(phrase, str):
        return ""
    p = " ".join(phrase.strip().split())
    if not p:
        return ""

    # Cut off common prepositional/object-introducing tails (drop direct objects / background PPs).
    cut_markers = [
        " with ", " into ", " onto ", " on ", " in ", " at ", " by ", " for ", " from ", " over ", " under ",
        " to ", " of ",
    ]
    lower_p = p.lower()
    cut_pos = None
    for m in cut_markers:
        idx = lower_p.find(m)
        if idx != -1:
            if cut_pos is None or idx < cut_pos:
                cut_pos = idx
    if cut_pos is not None and cut_pos > 0:
        p = p[:cut_pos].strip()

    # Remove trailing determiners that may remain
    while p.lower().endswith((" a", " an", " the")):
        p = p.rsplit(" ", 1)[0].strip()

    # Ensure obj still present
    if obj.strip().lower() not in p.lower():
        return ""
    return p


def filter_phrases_literal(obj: str, phrases: list):
    """Mode A: keep only phrases that literally include the obj string (case-insensitive)."""
    if not isinstance(obj, str) or not obj.strip():
        return []
    needle = obj.strip().lower()
    # boundary-aware match to avoid cases like obj="guy" matching "guitar"
    # also works for multi-word objs ("drum set") by allowing flexible whitespace.
    escaped = re.escape(needle)
    escaped = escaped.replace("\\ ", r"\\s+")
    pat = re.compile(r"(?<![A-Za-z0-9_])" + escaped + r"(?![A-Za-z0-9_])", re.IGNORECASE)
    out = []
    for p in phrases if isinstance(phrases, list) else []:
        if not isinstance(p, str):
            continue
        if pat.search(p):
            out.append(p)
    return out


def parse_json_response(response: str):
    """Parse JSON from API response, handling markdown code blocks."""
    text = response.strip()
    # Remove markdown code blocks if present
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first line (```json or ```)
        lines = lines[1:]
        # Remove last line if it's ```
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    
    try:
        return json.loads(text), None
    except json.JSONDecodeError as e:
        return None, f"JSON parse error: {e}"


def _looks_like_action_unit(text: str) -> bool:
    """Heuristic filter: drop obvious action phrases when we want nouns only."""
    if not isinstance(text, str):
        return True
    t = text.strip().lower()
    if not t:
        return True
    # Disallow gerunds and common verb-led phrases.
    action_markers = (
        "ing ",  # e.g., "pouring sauce"
        "ing",   # e.g., "running"
    )
    # Single-token gerund
    if " " not in t and t.endswith("ing"):
        return True
    # Multi-token phrase starting with a gerund
    if " " in t and t.split(" ", 1)[0].endswith("ing"):
        return True
    # Other common verb/action tokens (non-exhaustive)
    verb_starters = {
        "put", "puts", "putting", "poured", "pour", "pours", "pouring",
        "play", "plays", "playing", "run", "runs", "running",
        "talk", "talks", "talking", "cook", "cooks", "cooking",
        "make", "makes", "making", "fold", "folds", "folding",
        "catch", "catches", "catching", "shoot", "shoots", "shooting",
        "climb", "climbs", "climbing", "slide", "slides", "sliding",
        "drive", "drives", "driving", "hit", "hits", "hitting",
        "decorate", "decorates", "decorating",
    }
    first = t.split(" ", 1)[0]
    if first in verb_starters:
        return True
    # If phrase contains a preposition pattern typical for actions
    if " into " in t or " over " in t or " on " in t or " under " in t:
        # This is a strong indicator of an action phrase like "pouring liquid into bag"
        if first in verb_starters or first.endswith("ing"):
            return True
    return False


def filter_to_nouns_only(units):
    """Filter out action-like units, keep only noun/compound-noun candidates."""
    if not isinstance(units, list):
        return []
    kept = []
    for u in units:
        if isinstance(u, str) and not _looks_like_action_unit(u):
            kept.append(u)
    return kept


def filter_nouns_in_captions(nouns: list, captions: list) -> list:
    """Filter nouns to keep only those that actually appear in at least one caption."""
    if not isinstance(nouns, list) or not isinstance(captions, list):
        return []
    
    # Build combined text from all captions (lowercase)
    all_text = " ".join(cap.lower() for _, cap in captions)
    
    kept = []
    for noun in nouns:
        if not isinstance(noun, str) or not noun.strip():
            continue
        if noun.strip().lower() in all_text:
            kept.append(noun)
    return kept


def validate_result(result: dict, captions: list):
    """
    Validate extraction result. Check if any sentence has empty extraction.
    Returns (is_valid, empty_sentences) where empty_sentences is a list of (sent_id, caption) tuples.
    """
    if not result or "sentence_objs" not in result:
        return False, [(sid, cap) for sid, cap in captions]
    
    sentence_objs = result.get("sentence_objs", {})
    empty_sentences = []
    
    for sent_id, caption in captions:
        objs = sentence_objs.get(sent_id, [])
        if not objs:
            empty_sentences.append((sent_id, caption))
    
    # Consider valid if less than 20% are empty (some captions are genuinely meaningless)
    empty_ratio = len(empty_sentences) / len(captions) if captions else 0
    is_valid = empty_ratio < 0.2
    
    return is_valid, empty_sentences


def process_videos(
    videos,
    keys,
    output_dir,
    start_from=0,
    max_retries=3,
    phrase_mode="A",
    max_phrases_per_obj=8,
):
    """Process all videos and save results with robustness checks."""
    os.makedirs(output_dir, exist_ok=True)
    
    key_index = 0
    video_ids = list(videos.keys())
    total = len(video_ids)
    failed_videos = []  # Backward compatible: (video_id, empty_count_or_-1, total_caps)
    failed_details = []  # Detailed: dict records
    
    for i, video_id in enumerate(video_ids):
        if i < start_from:
            continue
            
        output_file = os.path.join(output_dir, f"{video_id}.json")
        
        # Skip if already processed
        if os.path.exists(output_file):
            print(f"[{i+1}/{total}] {video_id}: already exists, skipping")
            continue
        
        captions = videos[video_id]
        caption_block = build_caption_block(captions)
        
        # Retry logic with validation
        best_result = None
        best_empty_count = float('inf')
        last_api_error = None
        last_parse_error = None
        parse_failures = 0
        api_failures = 0
        
        for retry in range(max_retries):
            key = keys[key_index]
            retry_str = f"(retry {retry+1}/{max_retries})" if retry > 0 else ""
            print(f"[{i+1}/{total}] {video_id}: processing {retry_str}...")
            
            response, error = call_api(key, caption_block)
            
            if error:
                print(f"  API error: {error}, trying next key...")
                key_index = (key_index + 1) % len(keys)
                last_api_error = error
                api_failures += 1
                continue
            
            result, parse_error = parse_json_response(response)
            
            if parse_error:
                print(f"  Parse error: {parse_error}")
                key_index = (key_index + 1) % len(keys)
                last_parse_error = parse_error
                parse_failures += 1
                continue

            # Nouns-only post-filtering (defensive)
            if isinstance(result, dict):
                # if "sentence_objs" in result and isinstance(result["sentence_objs"], dict):
                #     for sid, units in list(result["sentence_objs"].items()):
                #         result["sentence_objs"][sid] = filter_to_nouns_only(units)
                if "video_objs" in result and isinstance(result["video_objs"], list):
                    result["video_objs"] = filter_to_nouns_only(result["video_objs"])
                    # Filter out nouns that don't actually appear in captions (prevent hallucinations)
                    result["video_objs"] = filter_nouns_in_captions(result["video_objs"], captions)
            
            # Validate result
            is_valid, empty_sentences = validate_result(result, captions)
            
            if not empty_sentences:
                # Perfect result, no empty extractions
                best_result = result
                print(f"  ✓ All {len(captions)} sentences extracted successfully")
                break
            
            # Track best result (least empty sentences)
            if len(empty_sentences) < best_empty_count:
                best_empty_count = len(empty_sentences)
                best_result = result
            
            if is_valid:
                # Acceptable result (< 20% empty)
                print(f"  ⚠ {len(empty_sentences)}/{len(captions)} sentences empty (acceptable):")
                for sid, cap in empty_sentences[:3]:  # Show first 3
                    print(f"    - [{sid}]: {cap[:60]}..." if len(cap) > 60 else f"    - [{sid}]: {cap}")
                break
            else:
                # Too many empty, retry
                print(f"  ✗ {len(empty_sentences)}/{len(captions)} sentences empty, retrying...")
                for sid, cap in empty_sentences[:5]:  # Show first 5
                    print(f"    - [{sid}]: {cap[:60]}..." if len(cap) > 60 else f"    - [{sid}]: {cap}")
                time.sleep(1)  # Small delay before retry
        
        # Save best result
        if best_result:
            # Deduplicate sentence_objs entries while preserving order
            # if "sentence_objs" in best_result and isinstance(best_result["sentence_objs"], dict):
            #     for sid, units in list(best_result["sentence_objs"].items()):
            #         if not isinstance(units, list):
            #             continue
            #         seen_units = set()
            #         unique_units = []
            #         for u in units:
            #             if u not in seen_units:
            #                 seen_units.add(u)
            #                 unique_units.append(u)
            #         best_result["sentence_objs"][sid] = unique_units

            # Deduplicate video_objs while preserving order
            if "video_objs" in best_result:
                seen = set()
                unique_objs = []
                for obj in best_result["video_objs"]:
                    if obj not in seen:
                        seen.add(obj)
                        unique_objs.append(obj)
                best_result["video_objs"] = unique_objs

            # Second stage: build minimal phrase set for each video object
            # NOTE: Stage2 now returns phrases (may include modifiers/actions) and embeds them directly
            # under video_objs as a dict: {obj: [phrases...]}
            video_objs_map = {}  # Only populated with non-empty phrase lists
            if best_result.get("video_objs"):
                mapping_response = None
                mapping_error = None
                for _ in range(len(keys)):
                    key = keys[key_index]
                    mapping_response, mapping_error = call_obj_phrases_api(
                        key=key,
                        video_objs=best_result.get("video_objs", []),
                        caption_block=caption_block,
                        phrase_mode=phrase_mode,
                        max_phrases_per_obj=max_phrases_per_obj,
                        max_retries=3,
                    )
                    if mapping_error:
                        key_index = (key_index + 1) % len(keys)
                        continue
                    parsed_map, parse_err = parse_obj_phrases_response(mapping_response)
                    if parse_err:
                        mapping_error = parse_err
                        key_index = (key_index + 1) % len(keys)
                        continue

                    # Dedup per object (phrases may include actions/modifiers, so do NOT noun-filter here)
                    cleaned = {}
                    for obj, phrases in parsed_map.items():
                        deduped = dedup_preserve_order(phrases)
                        if phrase_mode.upper() == "A":
                            deduped = filter_phrases_literal(obj, deduped)
                            trimmed = []
                            for ph in deduped:
                                all_objs = best_result.get("video_objs", [])
                                t = trim_phrase_remove_other_objs(obj, ph, all_objs)
                                if not t:
                                    t = fallback_extract_obj_patient_phrase(obj, ph, all_objs)
                                if t:
                                    t = trim_phrase_single_object(obj, t)
                                if t:
                                    trimmed.append(t)
                            deduped = dedup_preserve_order(trimmed)
                        cleaned[obj] = deduped

                    # Ensure all objs exist as keys, but only keep non-empty ones
                    for o in best_result.get("video_objs", []):
                        phrases = cleaned.get(o, [])
                        if phrases:  # Only keep objects with non-empty phrase lists
                            video_objs_map[o] = phrases

                    mapping_error = None
                    break

                if mapping_error:
                    # If mapping fails, keep empty lists but record failure details
                    failed_details.append({
                        "video_id": video_id,
                        "status": "OBJ_PHRASES_FAIL",
                        "empty": None,
                        "total": len(captions),
                        "api_failures": None,
                        "parse_failures": None,
                        "last_api_error": mapping_error,
                        "last_parse_error": None,
                    })

            # Only keep video-level annotations in output
            video_level_result = {
                "video_objs": video_objs_map,
            }

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(video_level_result, f, indent=2, ensure_ascii=False)
            
            # Create and save multi-hot noun vector
            nouns = list(video_objs_map.keys())
            tokenizer = get_tokenizer()
            noun_vector = create_noun_vector(nouns, tokenizer)
            vector_dir = os.path.join(os.path.dirname(output_dir), "noun_vectors", os.path.basename(output_dir))
            os.makedirs(vector_dir, exist_ok=True)
            vector_file = os.path.join(vector_dir, f"{video_id}.npy")
            np.save(vector_file, noun_vector)
            
            print(f"  -> saved to {output_file} ({len(nouns)} nouns, {int(noun_vector.sum())} tokens)")
            
            # Record if validation failed after all retries
            if best_empty_count > len(captions) * 0.2:
                failed_videos.append((video_id, best_empty_count, len(captions)))
                failed_details.append({
                    "video_id": video_id,
                    "status": "VALIDATION_FAIL",
                    "empty": int(best_empty_count) if best_empty_count != float('inf') else None,
                    "total": len(captions),
                    "api_failures": api_failures,
                    "parse_failures": parse_failures,
                    "last_api_error": last_api_error,
                    "last_parse_error": last_parse_error,
                })
        else:
            print(f"  FAILED: could not get valid result for {video_id}")
            failed_videos.append((video_id, -1, len(captions)))
            # Distinguish failure reasons
            if api_failures and not parse_failures:
                status = "API_FAIL"
                detail = last_api_error
            elif parse_failures and not api_failures:
                status = "PARSE_FAIL"
                detail = last_parse_error
            elif api_failures and parse_failures:
                status = "API_AND_PARSE_FAIL"
                detail = f"api={last_api_error}; parse={last_parse_error}"
            else:
                status = "UNKNOWN_FAIL"
                detail = None
            failed_details.append({
                "video_id": video_id,
                "status": status,
                "empty": None,
                "total": len(captions),
                "api_failures": api_failures,
                "parse_failures": parse_failures,
                "last_api_error": last_api_error,
                "last_parse_error": last_parse_error,
                "detail": detail,
            })
        
        # Small delay to avoid rate limiting
        time.sleep(0.5)
    
    # Summary
    print(f"\n" + "="*60)
    print(f"Done! Processed {total} videos.")
    if failed_videos:
        print(f"\n⚠ {len(failed_videos)} videos had issues:")
        for vid, empty, total_cap in failed_videos:
            if empty == -1:
                print(f"  - {vid}: complete failure")
            else:
                print(f"  - {vid}: {empty}/{total_cap} empty sentences")
        # Save failed list
        failed_file = os.path.join(output_dir, "_failed.txt")
        with open(failed_file, "w") as f:
            for vid, empty, total_cap in failed_videos:
                f.write(f"{vid}\t{empty}\t{total_cap}\n")
        print(f"\nFailed list saved to {failed_file}")

        # Save detailed failed list
        failed_detail_file = os.path.join(output_dir, "_failed_detail.tsv")
        with open(failed_detail_file, "w", encoding="utf-8") as f:
            f.write("video_id\tstatus\tempty\ttotal\tapi_failures\tparse_failures\tlast_api_error\tlast_parse_error\n")
            for r in failed_details:
                f.write(
                    f"{r.get('video_id')}\t{r.get('status')}\t{r.get('empty')}\t{r.get('total')}\t"
                    f"{r.get('api_failures')}\t{r.get('parse_failures')}\t{r.get('last_api_error')}\t{r.get('last_parse_error')}\n"
                )
        print(f"Failed detail saved to {failed_detail_file}")


def process_single_video(
    video_id: str,
    captions: list,
    key: str,
    output_dir: str,
    phrase_mode: str,
    max_phrases_per_obj: int,
    max_retries: int = 3,
):
    """Process a single video with a dedicated API key. Returns (video_id, status, detail)."""
    output_file = os.path.join(output_dir, f"{video_id}.json")
    
    # Skip if already processed
    if os.path.exists(output_file):
        return video_id, "skipped", None
    
    caption_block = build_caption_block(captions)
    
    # Stage 1: Extract nouns with retries
    best_result = None
    best_empty_count = float('inf')
    
    for retry in range(max_retries):
        response, error = call_api(key, caption_block)
        if error:
            time.sleep(0.5)
            continue
        
        result, parse_error = parse_json_response(response)
        if parse_error:
            time.sleep(0.5)
            continue
        
        # Nouns-only post-filtering
        if isinstance(result, dict):
            if "video_objs" in result and isinstance(result["video_objs"], list):
                result["video_objs"] = filter_to_nouns_only(result["video_objs"])
                result["video_objs"] = filter_nouns_in_captions(result["video_objs"], captions)
        
        is_valid, empty_sentences = validate_result(result, captions)
        
        if not empty_sentences:
            best_result = result
            break
        
        if len(empty_sentences) < best_empty_count:
            best_empty_count = len(empty_sentences)
            best_result = result
        
        if is_valid:
            break
        
        time.sleep(0.5)
    
    if not best_result:
        return video_id, "failed", "No valid result after retries"
    
    # Deduplicate video_objs
    if "video_objs" in best_result:
        seen = set()
        unique_objs = []
        for obj in best_result["video_objs"]:
            if obj not in seen:
                seen.add(obj)
                unique_objs.append(obj)
        best_result["video_objs"] = unique_objs
    
    # Stage 2: Get phrases for each object
    video_objs_map = {}
    if best_result.get("video_objs"):
        for _ in range(max_retries):
            mapping_response, mapping_error = call_obj_phrases_api(
                key=key,
                video_objs=best_result.get("video_objs", []),
                caption_block=caption_block,
                phrase_mode=phrase_mode,
                max_phrases_per_obj=max_phrases_per_obj,
                max_retries=1,
            )
            if mapping_error:
                time.sleep(0.5)
                continue
            
            parsed_map, parse_err = parse_obj_phrases_response(mapping_response)
            if parse_err:
                time.sleep(0.5)
                continue
            
            # Clean up phrases
            cleaned = {}
            for obj, phrases in parsed_map.items():
                deduped = dedup_preserve_order(phrases)
                if phrase_mode.upper() == "A":
                    deduped = filter_phrases_literal(obj, deduped)
                    trimmed = []
                    for ph in deduped:
                        all_objs = best_result.get("video_objs", [])
                        t = trim_phrase_remove_other_objs(obj, ph, all_objs)
                        if not t:
                            t = fallback_extract_obj_patient_phrase(obj, ph, all_objs)
                        if t:
                            t = trim_phrase_single_object(obj, t)
                        if t:
                            trimmed.append(t)
                    deduped = dedup_preserve_order(trimmed)
                cleaned[obj] = deduped
            
            for o in best_result.get("video_objs", []):
                phrases = cleaned.get(o, [])
                if phrases:
                    video_objs_map[o] = phrases
            break
    
    # Save JSON result
    video_level_result = {"video_objs": video_objs_map}
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(video_level_result, f, indent=2, ensure_ascii=False)
    
    # Create and save multi-hot noun vector
    nouns = list(video_objs_map.keys())
    tokenizer = get_tokenizer()
    noun_vector = create_noun_vector(nouns, tokenizer)
    vector_dir = os.path.join(os.path.dirname(output_dir), "noun_vectors", os.path.basename(output_dir))
    os.makedirs(vector_dir, exist_ok=True)
    vector_file = os.path.join(vector_dir, f"{video_id}.npy")
    np.save(vector_file, noun_vector)
    
    return video_id, "success", len(video_objs_map)


def process_videos_parallel(
    videos,
    keys,
    output_dir,
    phrase_mode="A",
    max_phrases_per_obj=8,
    max_workers=None,
):
    """Process videos in parallel using multiple API keys."""
    os.makedirs(output_dir, exist_ok=True)
    
    video_ids = list(videos.keys())
    total = len(video_ids)
    num_keys = len(keys)
    
    if max_workers is None:
        max_workers = min(num_keys, total)
    
    print(f"\n{'='*60}")
    print(f"Parallel processing: {total} videos with {max_workers} workers ({num_keys} keys)")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")
    
    # Pre-load tokenizer to avoid race conditions
    print("Pre-loading tokenizer...")
    get_tokenizer()
    
    # Progress tracking (thread-safe)
    progress_lock = threading.Lock()
    progress = {"done": 0, "success": 0, "skipped": 0, "failed": 0}
    start_time = time.time()
    
    def update_progress(status, is_retry=False):
        with progress_lock:
            progress["done"] += 1
            if is_retry:
                # Retry: if success, failed-1 and success+1; if failed again, no change to failed
                if status == "success":
                    progress["success"] += 1
                    progress["failed"] -= 1
                # If still failed, don't increment failed again (already counted)
            else:
                # First round: normal counting
                progress[status] += 1
            
            done = progress["done"]
            elapsed = time.time() - start_time
            speed = done / elapsed if elapsed > 0 else 0
            if done % 5 == 0 or done == total:
                print(f"[{done}/{total}] success={progress['success']}, skipped={progress['skipped']}, failed={progress['failed']} | {speed:.1f} vid/s")
    
    failed_videos = []
    failed_lock = threading.Lock()
    
    def run_batch(video_list, is_retry=False):
        """Run a batch of videos, return list of failed (video_id, captions, detail)."""
        batch_failed = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for i, (video_id, captions) in enumerate(video_list):
                key = keys[i % num_keys]
                future = executor.submit(
                    process_single_video,
                    video_id,
                    captions,
                    key,
                    output_dir,
                    phrase_mode,
                    max_phrases_per_obj,
                )
                futures[future] = (video_id, captions)
            
            for future in as_completed(futures):
                video_id, captions = futures[future]
                try:
                    vid, status, detail = future.result()
                    update_progress(status, is_retry=is_retry)
                    if status == "failed":
                        with failed_lock:
                            batch_failed.append((vid, captions, detail))
                except Exception as e:
                    update_progress("failed", is_retry=is_retry)
                    with failed_lock:
                        batch_failed.append((video_id, captions, str(e)))
        return batch_failed
    
    # First round
    video_list = [(vid, videos[vid]) for vid in video_ids]
    round1_failed = run_batch(video_list, is_retry=False)
    
    # Retry failed videos (up to 3 more rounds)
    all_failed = round1_failed
    for retry_round in range(1, 4):
        if not all_failed:
            break
        print(f"\n--- Retry round {retry_round}: {len(all_failed)} failed videos ---")
        retry_list = [(vid, caps) for vid, caps, _ in all_failed]
        all_failed = run_batch(retry_list, is_retry=True)
    
    # Collect final failures
    for vid, caps, detail in all_failed:
        failed_videos.append((vid, detail))
    
    # Summary
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Done! Processed {total} videos in {elapsed:.1f}s ({total/elapsed:.1f} vid/s)")
    print(f"  Success: {progress['success']}")
    print(f"  Skipped: {progress['skipped']}")
    print(f"  Failed: {len(failed_videos)} (after retries)")
    print(f"{'='*60}")
    
    if failed_videos:
        failed_file = os.path.join(output_dir, "_failed.txt")
        with open(failed_file, "w") as f:
            for vid, detail in failed_videos:
                f.write(f"{vid}\t{detail}\n")
        print(f"\nFailed list saved to {failed_file}")
    
    return progress


def process_all_splits(
    keys,
    base_dir=".",
    phrase_mode="A",
    max_phrases_per_obj=8,
    parallel=True,
    max_workers=None,
):
    """Process all train/val/test splits."""
    splits = ["train", "val", "test"]
    results = {}
    
    for split in splits:
        input_file = os.path.join(base_dir, f"{split}_preprocessed.txt")
        output_dir = os.path.join(base_dir, "annotations", "nouns", split)
        
        if not os.path.exists(input_file):
            print(f"\n[SKIP] {input_file} not found")
            continue
        
        print(f"\n{'#'*60}")
        print(f"# Processing {split.upper()} split")
        print(f"# Input: {input_file}")
        print(f"# Output: {output_dir}")
        print(f"{'#'*60}")
        
        videos = load_captions(input_file)
        print(f"Loaded {len(videos)} videos from {input_file}")
        
        if parallel:
            result = process_videos_parallel(
                videos, keys, output_dir,
                phrase_mode=phrase_mode,
                max_phrases_per_obj=max_phrases_per_obj,
                max_workers=max_workers,
            )
        else:
            process_videos(
                videos, keys, output_dir,
                phrase_mode=phrase_mode,
                max_phrases_per_obj=max_phrases_per_obj,
            )
            result = {"done": len(videos)}
        
        results[split] = result
    
    # Final summary
    print(f"\n{'#'*60}")
    print("# ALL SPLITS COMPLETED")
    print(f"{'#'*60}")
    for split, res in results.items():
        print(f"  {split}: {res}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Extract visual atomic units from captions")
    parser.add_argument("--input", "-i", default="val_preprocessed.txt", help="Input file path")
    parser.add_argument("--output", "-o", default="annotations/nouns/val", help="Output directory")
    parser.add_argument("--keys", "-k", default=".keys", help="API keys file")
    parser.add_argument("--start", "-s", type=int, default=0, help="Start from video index (for resuming)")
    parser.add_argument(
        "--phrase_mode",
        choices=["A", "B", "a", "b"],
        default="A",
        help="Stage2 phrase mapping mode: A=strict literal contains obj (default); B=semantic/coreference",
    )
    parser.add_argument(
        "--max_phrases_per_obj",
        type=int,
        default=8,
        help="Max phrases to keep per object in stage2 mapping",
    )
    parser.add_argument(
        "--parallel", "-p",
        action="store_true",
        help="Enable parallel processing with multiple API keys",
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=None,
        help="Number of parallel workers (default: number of keys)",
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Process all splits (train/val/test) automatically",
    )
    args = parser.parse_args()
    
    # Load keys
    keys = load_keys(args.keys)
    print(f"Loaded {len(keys)} API key(s)")
    
    # Process all splits
    if args.all:
        process_all_splits(
            keys,
            base_dir=os.path.dirname(args.input) or ".",
            phrase_mode=args.phrase_mode,
            max_phrases_per_obj=args.max_phrases_per_obj,
            parallel=args.parallel,
            max_workers=args.workers,
        )
        return
    
    # Load captions for single file
    videos = load_captions(args.input)
    print(f"Loaded {len(videos)} videos from {args.input}")
    
    # Process
    if args.parallel:
        process_videos_parallel(
            videos,
            keys,
            args.output,
            phrase_mode=args.phrase_mode,
            max_phrases_per_obj=args.max_phrases_per_obj,
            max_workers=args.workers,
        )
    else:
        process_videos(
            videos,
            keys,
            args.output,
            start_from=args.start,
            phrase_mode=args.phrase_mode,
            max_phrases_per_obj=args.max_phrases_per_obj,
        )


if __name__ == "__main__":
    main()
