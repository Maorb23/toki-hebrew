"""Benchmark a conservative Hebrew spelling checker using Dicta + SymSpellPy.

Dependencies: uv pip install symspellpy datasets huggingface_hub duckdb
Dictionary: he-100k.txt from the official SymSpell repository.

This detects suspected *spelling* errors, not grammatical agreement errors.
"""

import argparse
import importlib.util
import re
import time
from pathlib import Path

from huggingface_hub import hf_hub_download
from symspellpy import SymSpell, Verbosity

HERE = Path(__file__).resolve().parent
HEBREW_WORD = re.compile(r"[\u0590-\u05FF]+")
PREFIXES = ("ה", "ו", "ב", "כ", "ל", "מ", "ש")

# input, desired spelling (not a grammar-correction target)
CASES = [
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
    ("הילדים הולכות לבית הספר", "הילדים הולכות לבית הספר"),
    ("הילדה אוכל תפוח", "הילדה אוכל תפוח"),
    ("הילד הולך הביתה", "הילד הולך הביתה"),
    ("הילדות משחקים בחצר", "הילדות משחקים בחצר"),
]


def load_models(dictionary: Path):
    start = time.perf_counter()
    spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
    if not spell.load_dictionary(str(dictionary), term_index=0, count_index=1, encoding="utf-8"):
        raise RuntimeError(f"Unable to load dictionary: {dictionary}")

    path = hf_hub_download(
        repo_id="dicta-il/wordlist", filename="wordlist_query.py", repo_type="dataset"
    )
    spec = importlib.util.spec_from_file_location("dicta_wordlist_query", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    lexicon = module.WordlistQuery(genre="Modern", disk=str(HERE / "wordlist-index.duckdb"))
    return spell, lexicon, (time.perf_counter() - start) * 1000


class Checker:
    def __init__(self, spell, lexicon):
        self.spell = spell
        self.lexicon = lexicon
        self.cache = {}

    def modern_anchor(self, word):
        if word not in self.cache:
            matches = self.lexicon.get_word_options(word)
            self.cache[word] = any(
                match.morph_anchor and "Modern" in match.anchor_genres
                for match in matches
            )
        return self.cache[word]

    def candidates(self, word):
        # Generate suggestions from the *whole word* first.
        results = {}
        for s in self.spell.lookup(word, Verbosity.ALL, max_edit_distance=2):
            if s.term != word and self.modern_anchor(s.term):
                results[s.term] = (s.distance, 0, s.count)

        # Hebrew prefixes are often missing from the frequency dictionary.
        # Preserve one initial prefix, correct the remainder, then validate
        # the reconstructed full form against Dicta.
        if len(word) > 3 and word[0] in PREFIXES:
            prefix, stem = word[0], word[1:]
            for s in self.spell.lookup(stem, Verbosity.ALL, max_edit_distance=2):
                full = prefix + s.term
                if full != word and self.modern_anchor(full):
                    score = (s.distance, 1, s.count)
                    previous = results.get(full)
                    if previous is None or (score[0], -score[2]) < (previous[0], -previous[2]):
                        results[full] = score

        # Prefer fewer edits, then retained prefix, then frequency. No
        # contextual language model is used, so results remain provisional.
        ranked = sorted(
            results.items(),
            key=lambda item: (item[1][0], -item[1][1], -item[1][2]),
        )
        return [term for term, _ in ranked[:5]]

    def correct(self, sentence):
        issues = []
        def replace(match):
            word = match.group(0)
            if self.modern_anchor(word):
                return word
            options = self.candidates(word)
            issues.append((word, options))
            return options[0] if options else word

        output = HEBREW_WORD.sub(replace, sentence)
        return output, issues


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", help="Single custom Hebrew sentence")
    parser.add_argument("--dictionary", default="he-100k.txt")
    args = parser.parse_args()
    dictionary = Path(args.dictionary)
    if not dictionary.is_file():
        parser.error(f"Dictionary not found: {dictionary}. Place he-100k.txt in this directory.")

    spell, lexicon, init_ms = load_models(dictionary)
    checker = Checker(spell, lexicon)
    print(f"Initialization: {init_ms:.1f} ms")
    tests = [(args.text, None)] if args.text else CASES
    times, exact = [], 0
    try:
        for original, expected in tests:
            start = time.perf_counter()
            predicted, issues = checker.correct(original)
            ms = (time.perf_counter() - start) * 1000
            times.append(ms)
            print(f"\nInput:      {original}")
            print(f"Suggestion: {predicted}")
            print(f"Status:     {'SUSPICIOUS' if issues else 'NO SPELLING ISSUE DETECTED'}")
            for word, options in issues:
                print(f"  {word}: {options or 'no validated suggestions'}")
            print(f"Warm lookup: {ms:.2f} ms")
            if expected is not None:
                ok = predicted == expected
                exact += ok
                print(f"Exact match: {'PASS' if ok else 'FAIL'} | expected: {expected}")
        print(f"\n{'='*60}")
        print(f"Sentences: {len(tests)}")
        print(f"Initialization: {init_ms:.1f} ms")
        print(f"Mean warm sentence lookup: {sum(times)/len(times):.2f} ms")
        if not args.text:
            print(f"Exact sentence matches: {exact}/{len(tests)}")
        print("Note: Warm timings include Dicta dictionary queries; cached words may be faster.")
    finally:
        lexicon.close()


if __name__ == "__main__":
    main()

