#!/usr/bin/env python3
"""Fast Hebrew spellchecking benchmark: Dicta attested words + SymSpell.

Install into your existing environment:
    uv pip install symspellpy datasets

Keep he-100k.txt in the same directory and run:
    python test_dicta_symspell_fast.py
    python test_dicta_symspell_fast.py --text 'אנכנו הולכים לבית הספר'
    python test_dicta_symspell_fast.py --corpus my_hebrew_corpus.txt

What this DOES:
  * Detect words not observed in Dicta's Modern Hebrew attested lexicon.
  * Suggest replacement spellings using SymSpell, preserving prefixes where possible.
  * Validate candidates with fast in-memory set membership.
  * Optionally use bigram counts from a separate, unlabeled Hebrew corpus for ranking.

What this DOES NOT:
  * Check grammatical agreement (e.g. הילדים הולכות).
  * Guarantee an unchanged word is correctly spelled.
  * Claim contextual knowledge without a separate corpus.

Dicta wordlist: https://huggingface.co/datasets/dicta-il/wordlist
Dicta dataset license: CC BY-NC 4.0 (check terms before commercial use).
"""

import argparse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import math
import re
import statistics
import time

from datasets import load_dataset
from symspellpy import SymSpell, Verbosity

ROOT = Path(__file__).resolve().parent
HEBREW_WORD = re.compile(r"[\u05D0-\u05EA]+")
PREFIXES = "והבכלמש"  # Only attempt a one-letter prefix fallback, not morphology rules.

# These are evaluation examples only; never used to train or tune ranking.

CASES = [
    # 1. Original spelling tests
    ("אנחנו הולכים לבית הספר", "אנחנו הולכים לבית הספר"),
    ("אנכנו הולכים לבית הספר", "אנחנו הולכים לבית הספר"),
    ("הילדים הולכים לבית הספר", "הילדים הולכים לבית הספר"),
    ("הילידם הולכים לבית הספר", "הילדים הולכים לבית הספר"),
    ("הילדה אוכלת תפוח", "הילדה אוכלת תפוח"),
    ("הילדה אוכלת תפו", "הילדה אוכלת תפוח"),
    ("אני רוצה ללמוד עברית", "אני רוצה ללמוד עברית"),
    ("אני רוצ ללמוד עברית", "אני רוצה ללמוד עברית"),
    ("היום מזג האוויר נעים", "היום מזג האוויר נעים"),
    ("היופ מזג האוויר נעים", "היום מזג האוויר נעים"),
    ("היא עובדת במשרד", "היא עובדת במשרד"),
    ("היא עובדת במישרד", "היא עובדת במשרד"),
    ("הספר נמצא על השולחן", "הספר נמצא על השולחן"),
    ("הספר נימצא על השולחן", "הספר נמצא על השולחן"),
    ("מחר ניפגש בתל אביב", "מחר ניפגש בתל אביב"),
    ("מחר נפיגש בתל אביב", "מחר ניפגש בתל אביב"),

    # 2. Correct words that must be preserved
    ("אני אוהב לקרוא ספרים", "אני אוהב לקרוא ספרים"),
    ("אני אוהב לקרוע ספרים", "אני אוהב לקרוע ספרים"),
    ("הילדים משחקים בכדור", "הילדים משחקים בכדור"),
    ("הילדים משחקים בכדורסל", "הילדים משחקים בכדורסל"),
    ("היא עובדת במשרד החדש", "היא עובדת במשרד החדש"),
    ("למרות הגשם יצאנו לטייל", "למרות הגשם יצאנו לטייל"),

    # 3. More difficult spelling errors
    ("אני מתכוון ללכת מחר", "אני מתכוון ללכת מחר"),
    ("אני מתכבן ללכת מחר", "אני מתכוון ללכת מחר"),
    ("החלטנו להיפגש בערב", "החלטנו להיפגש בערב"),
    ("החלתנו להיפגש בערב", "החלטנו להיפגש בערב"),
    ("הוא ביקש ממני עזרה", "הוא ביקש ממני עזרה"),
    ("הוא ביקש ממני אזרה", "הוא ביקש ממני עזרה"),
    ("המחשב הפסיק לעבוד", "המחשב הפסיק לעבוד"),
    ("המחשף הפסיק לעבוד", "המחשב הפסיק לעבוד"),

    # 4. Multiple spelling errors in one sentence
    ("אנכנו רוצ ללמוד עיברית", "אנחנו רוצים ללמוד עברית"),
    ("הילידם משחקם בחצר", "הילדים משחקים בחצר"),
    ("המורה ביקשה לפתוח את הספר", "המורה ביקשה לפתוח את הספר"),
    ("המורה ביקשא לפטוח את הסבר", "המורה ביקשה לפתוח את הספר"),

    # 5. Grammatical agreement errors
    # These are NOT expected to be fixed by spelling-only SymSpell.
    ("הילדים הולכות לבית הספר", "הילדים הולכים לבית הספר"),
    ("הילדה אוכל תפוח", "הילדה אוכלת תפוח"),
    ("הילד הולך הביתה", "הילד הולך הביתה"),
    ("הילדות משחקים בחצר", "הילדות משחקות בחצר"),
    ("הילדים שמחה מאוד", "הילדים שמחים מאוד"),
    ("אני חושב שמלחמות נכונה", "אני חושב שמלחמות נכונות"),

    # 6. Context-dependent real-word mistakes
    # All individual words may exist in the dictionary.
    ("אני רוצה לקרוא ספרים", "אני רוצה לקרוא ספרים"),
    ("אני רוצה לקרוע ספרים", "אני רוצה לקרוע ספרים"),
    ("אני רוצה לשתות מים", "אני רוצה לשתות מים"),
    ("אני רוצה לשטות מים", "אני רוצה לשתות מים"),
]



@dataclass(frozen=True)
class Candidate:
    word: str
    edit_distance: int
    frequency: int
    via_prefix: bool


# STEP 1: Read Dicta's *attested* wordforms, not 4 million hypothetical ones.
# Each row records a wordform and a prefix observed together in a corpus.
# Filtering the actual 'Modern' genre means we don't have to guess which
# combinations of prefixes are legal, and can check membership in O(1).
def load_modern_vocabulary() -> set[str]:
    ds = load_dataset("dicta-il/wordlist", "attested", split="train")
    genre_feature = ds.features["genres"].feature
    modern_id = genre_feature.str2int("Modern")
    valid = set()
    for row in ds.select_columns(["wordform", "prefix", "genres"]):
        if modern_id in row["genres"]:
            valid.add((row["prefix"] or "") + row["wordform"])
    return valid


# STEP 2: Load SymSpell's existing word-frequency dictionary.
# SymSpell provides edit-distance candidates; Dicta will decide whether each
# proposed surface word is an observed modern Hebrew word.
def load_symspell(path: Path) -> SymSpell:
    spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
    ok = spell.load_dictionary(str(path), 0, 1, encoding="utf-8")
    if not ok:
        raise RuntimeError(f"Could not load SymSpell dictionary: {path}")
    return spell


# STEP 3 (OPTIONAL): Read your own *separate*, unlabeled Hebrew text corpus.
# Neighboring-word frequencies provide a weak, real contextual signal.
# Do not train on CASES above: that would leak the test answers.
def load_bigram_counts(path: Path | None):
    if path is None:
        return Counter(), Counter()
    bigrams, left_totals = Counter(), Counter()
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            words = HEBREW_WORD.findall(line)
            for first, second in zip(words, words[1:]):
                bigrams[(first, second)] += 1
                left_totals[first] += 1
    return bigrams, left_totals


class HebrewSpellingChecker:
    def __init__(self, spell, modern_words, bigrams,
                 max_edits=2, max_candidates=10):
        self.spell = spell # SymSpell instance for edit-distance candidate generation.
        self.words = modern_words # Set of attested modern Hebrew words for fast membership checking.
        self.bigrams = bigrams # Counter of bigram frequencies for optional context-based ranking.
        self.max_edits = max_edits # Maximum edit distance for candidate generation.
        self.max_candidates = max_candidates # Maximum number of candidates to return for ranking.
        self.candidate_cache = {} # Cache for previously computed candidates to speed up repeated queries.

    # STEP 4: Ask SymSpell for edit-distance candidates, including a fallback
    # that fixes a stem while retaining an initial Hebrew prefix.
    # The fallback does NOT imply every prefix is grammatically permissible:
    # the complete resulting word MUST be present in Dicta's attested set.
    def candidates(self, word: str) -> list[Candidate]:
        if word in self.candidate_cache:
            return self.candidate_cache[word]

        found: dict[str, Candidate] = {}

        def offer(term, distance, count, via_prefix):
            
            if term == word or term not in self.words:
                return
            candidate = Candidate(term, distance, count, via_prefix)
            old = found.get(term)
            if old is None or (distance, -count) < (old.edit_distance, -old.frequency):
                found[term] = candidate

        for s in self.spell.lookup(word, Verbosity.ALL, max_edit_distance=self.max_edits):
            offer(s.term, s.distance, s.count, False)

        if len(word) >= 4 and word[0] in PREFIXES:
            prefix, stem = word[0], word[1:]
            for s in self.spell.lookup(stem, Verbosity.ALL, max_edit_distance=self.max_edits):
                offer(prefix + s.term, s.distance, s.count, True)

        # Rank a shortlist by edit distance, then source frequency.
        # Enlarging the SymSpell dictionary is optional because generating its
        # delete index for hundreds of thousands of words costs RAM/startup.
        shortlist = sorted(
            found.values(),
            key=lambda c: (c.edit_distance, -c.frequency, c.word),
        )[:self.max_candidates]
        self.candidate_cache[word] = shortlist
        return shortlist

    # STEP 5: Rank with the optional preceding/following-word context.
    # Without --corpus this explicitly falls back to distance + frequency.
    # No supervised grammatical model or invented context score is used.
    def rank(self, word: str, previous: str | None, following: str | None):
        choices = self.candidates(word)
        if not choices:
            return []

        def score(c: Candidate):
            # Smaller edit distance is strongly preferred, but not absolute.
            value = -2.0 * c.edit_distance + 0.25 * math.log1p(c.frequency)
            if self.bigrams:
                # Add-one smoothed bigram evidence; uniform unseen pairs don't
                # create fake distinctions between unknown candidate contexts.
                if previous is not None:
                    hits = self.bigrams[(previous, c.word)]
                    if hits:
                        value += 1.5 * math.log1p(hits)
                if following is not None:
                    hits = self.bigrams[(c.word, following)]
                    if hits:
                        value += 1.5 * math.log1p(hits)
            return value

        return sorted(choices, key=lambda c: (-score(c), c.edit_distance, c.word))

    # STEP 6: Leave observed Modern forms unchanged; only change words for
    # which a validated correction exists. Flag all other unknown words.
    # Preserve punctuation and spaces by editing only regex word spans.
    def correct(self, sentence: str):
        matches = list(HEBREW_WORD.finditer(sentence))
        tokens = [m.group() for m in matches]
        replacements = []
        issues = []
        t_generate = 0.0
        t_rank = 0.0

        for i, word in enumerate(tokens):
            if word in self.words:
                replacements.append(word)
                continue

            previous = tokens[i - 1] if i else None
            following = tokens[i + 1] if i + 1 < len(tokens) else None
            start = time.perf_counter()
            candidate_list = self.candidates(word)
            t_generate += time.perf_counter() - start
            start = time.perf_counter()
            ranked = self.rank(word, previous, following)
            t_rank += time.perf_counter() - start

            replacement = ranked[0].word if ranked else word
            replacements.append(replacement)
            issues.append((word, replacement, [c.word for c in ranked[:5]]))

        pieces = []
        offset = 0
        for match, replacement in zip(matches, replacements):
            pieces.append(sentence[offset:match.start()])
            pieces.append(replacement)
            offset = match.end()
        pieces.append(sentence[offset:])
        return "".join(pieces), issues, t_generate * 1000, t_rank * 1000


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", help="Test one sentence instead of the 20 built-in cases")
    parser.add_argument("--dictionary", default=str(ROOT / "he-100k.txt"))
    parser.add_argument("--corpus", type=Path, help="Independent UTF-8 Hebrew corpus for bigrams")
    parser.add_argument("--distance", type=int, choices=[1, 2], default=2)
    parser.add_argument("--top", type=int, default=10, help="Max candidates ranked per word")
    args = parser.parse_args()

    dictionary = Path(args.dictionary)
    if not dictionary.is_file():
        parser.error(f"Missing dictionary: {dictionary}")
    if args.corpus is not None and not args.corpus.is_file():
        parser.error(f"Missing corpus: {args.corpus}")

    # STEP 7: Measure initialization separately from inference.
    start = time.perf_counter()
    modern_words = load_modern_vocabulary()
    vocab_ms = 1000 * (time.perf_counter() - start)
    start = time.perf_counter()
    spell = load_symspell(dictionary)
    symspell_ms = 1000 * (time.perf_counter() - start)
    start = time.perf_counter()
    bigrams, left_totals = load_bigram_counts(args.corpus)
    corpus_ms = 1000 * (time.perf_counter() - start)

    checker = HebrewSpellingChecker(spell, modern_words, bigrams,
                                   max_edits=args.distance, max_candidates=args.top)
    print(f"Dicta Modern surface vocabulary: {len(modern_words):,} words ({vocab_ms:.1f} ms)")
    print(f"SymSpell dictionary load: {symspell_ms:.1f} ms")
    print(f"Bigram corpus: {sum(bigrams.values()):,} observed pairs ({corpus_ms:.1f} ms)")
    print("Ranking: " + ("distance + frequency + corpus bigrams" if bigrams else
                         "distance + frequency (no context corpus provided)"))

    # STEP 8: Measure each sentence; also repeat the batch to show the benefit
    # of caching candidates. This is not a cold-start inference measurement.
    cases = [(args.text, None)] if args.text else CASES
    times = []
    exact = 0
    for sentence, expected in cases:
        start = time.perf_counter()
        predicted, issues, generation_ms, rank_ms = checker.correct(sentence)
        elapsed_ms = (time.perf_counter() - start) * 1000
        times.append(elapsed_ms)
        print(f"\nInput:       {sentence}")
        print(f"Suggestion:  {predicted}")
        print(f"Status:      {'SUSPICIOUS' if issues else 'NO SPELLING ISSUE DETECTED'}")
        for word, replacement, suggestions in issues:
            print(f"  {word} -> {replacement} | candidates: {suggestions}")
        print(f"Latency:     {elapsed_ms:.3f} ms ",
              f"(generation {generation_ms:.3f}, rank {rank_ms:.3f})")
        if expected is not None:
            exact += predicted == expected
            print(f"Exact match: {'PASS' if predicted == expected else 'FAIL'} | expected: {expected}")

    repeated = []
    for sentence, _ in cases:
        start = time.perf_counter()
        checker.correct(sentence)
        repeated.append((time.perf_counter() - start) * 1000)

    print("\n" + "=" * 60)
    print(f"Sentences: {len(cases)}")
    print(f"First-pass mean: {statistics.mean(times):.3f} ms")
    print(f"Repeat-pass mean (cached candidates): {statistics.mean(repeated):.3f} ms")
    print(f"Repeat-pass median: {statistics.median(repeated):.3f} ms")
    if args.text is None:
        print(f"Exact sentence matches: {exact}/{len(cases)}")
    print("CAUTION: Bigram ranking requires an independent corpus; results are experimental.")


if __name__ == "__main__":
    main()

