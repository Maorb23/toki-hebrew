"""Benchmark native SymSpellPy correction on Hebrew text.

Setup:
    uv pip install symspellpy
    curl -L -o he-100k.txt \
      https://raw.githubusercontent.com/wolfgarbe/SymSpell/master/SymSpell.FrequencyDictionary/he-100k.txt

Run:
    python test_symspell_hebrew.py
    python test_symspell_hebrew.py --text 'אנכנו הולכים לביית הספר'
"""

import argparse
import time
from pathlib import Path

from symspellpy import SymSpell, Verbosity

# (sentence, expected spelling change?)
# These are evaluation labels, not hand-coded corrections.
TESTS = [
    ("אנחנו הולכים לבית הספר", False),
    ("אנכנו הולכים לבית הספר", True),
    ("הילדים הולכים לבית הספר", False),
    ("הילידם הולכים לבית הספר", True),
    ("הילדה אוכלת תפוח", False),
    ("הילדה אוכלת תפו", True),
    ("אני רוצה ללמוד עברית", False),
    ("אני רוצ ללמוד עברית", True),
    ("היום מזג האוויר נעים", False),
    ("היופ מזג האוויר נעים", True),
    ("היא עובדת במשרד", False),
    ("היא עובדת במישרד", True),
    ("הספר נמצא על השולחן", False),
    ("הספר נימצא על השולחן", True),
    ("מחר ניפגש בתל אביב", False),
    ("מחר נפיגש בתל אביב", True),
    ("הילדים הולכות לבית הספר", False),  # Grammar error; spelling is valid.
    ("הילדה אוכל תפוח", False),         # Grammar error; spelling is valid.
    ("הילד הולך הביתה", False),
    ("הילדות משחקים בחצר", False),      # Grammar error; spelling is valid.
]


def main():
    parser = argparse.ArgumentParser(description="Native SymSpellPy Hebrew benchmark")
    parser.add_argument("--text", help="Analyze just one sentence")
    parser.add_argument("--dictionary", default="he-100k.txt")
    parser.add_argument("--distance", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()

    dictionary = Path(args.dictionary)
    if not dictionary.is_file():
        parser.error(
            f"Missing dictionary: {dictionary}\n"
            "Download it with:\n"
            "curl -L -o he-100k.txt "
            "https://raw.githubusercontent.com/wolfgarbe/SymSpell/master/"
            "SymSpell.FrequencyDictionary/he-100k.txt"
        )

    start = time.perf_counter()
    spell = SymSpell(max_dictionary_edit_distance=args.distance, prefix_length=7)
    if not spell.load_dictionary(dictionary, term_index=0, count_index=1, encoding="utf-8"):
        parser.error(f"Could not load dictionary: {dictionary}")
    load_ms = (time.perf_counter() - start) * 1000
    print(f"Dictionary: {len(spell.words):,} words; load time: {load_ms:.1f} ms")

    tests = [(args.text, None)] if args.text else TESTS
    matched = 0
    elapsed = []

    for sentence, expected_change in tests:
        # Native SymSpell sentence-level correction. No manual Hebrew rules.
        start = time.perf_counter()
        suggestions = spell.lookup_compound(sentence, max_edit_distance=args.distance)
        duration_ms = (time.perf_counter() - start) * 1000
        elapsed.append(duration_ms)

        correction = suggestions[0].term if suggestions else sentence
        changed = correction != sentence

        print("\n" + "-" * 60)
        print(f"Input:       {sentence}")
        print(f"Suggestion:  {correction}")
        print(f"Changed:     {'YES' if changed else 'NO'}")
        print(f"Time:        {duration_ms:.2f} ms")
        if expected_change is not None:
            ok = changed == expected_change
            matched += ok
            print(f"Change test: {'PASS' if ok else 'FAIL'}")

        # Native single-token candidates, useful for inspecting overcorrection.
        if args.text:
            for word in sentence.split():
                options = spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=args.distance)
                print(f"  {word}: {', '.join(str(s.term) for s in options[:3]) or '(none)'}")

    print("\n" + "=" * 60)
    print(f"Sentences:          {len(tests)}")
    print(f"Dictionary load:    {load_ms:.1f} ms")
    print(f"Total lookup time:  {sum(elapsed):.1f} ms")
    print(f"Mean lookup time:   {sum(elapsed) / len(elapsed):.2f} ms")
    print(f"Median lookup time: {sorted(elapsed)[len(elapsed) // 2]:.2f} ms (upper median)")
    if not args.text:
        print(f"Change/no-change:   {matched}/{len(tests)} (NOT correction accuracy)")


if __name__ == "__main__":
    main()

