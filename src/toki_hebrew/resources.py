"""Local replaceable data loaders. Resources are shared by path, stat and edit radius."""
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import json
import re
from symspellpy import SymSpell

HEBREW_WORD = re.compile(r"[א-ת]+")
# Consume marks with letters so pointed words are never split into spelling fragments.
TOKEN = re.compile(r"[^\W\d_]+(?:[\u0591-\u05BD\u05BF-\u05C7]*[^\W\d_]*)*", re.UNICODE)


@dataclass(frozen=True)
class Resources:
    spell: SymSpell
    words: frozenset[str]
    bigrams: dict
    totals: dict
    vocabulary_size: int


def signature(path):
    p = Path(path).resolve()
    stat = p.stat()
    return str(p), stat.st_mtime_ns, stat.st_size


def corpus_counts(path):
    pairs = Counter()
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            words = HEBREW_WORD.findall(line)
            pairs.update(zip(words, words[1:]))
    return pairs


def write_bigram_index(corpus, destination):
    """Explicit offline preprocessing; resulting index inherits source restrictions."""
    pairs = corpus_counts(corpus)
    with Path(destination).open("w", encoding="utf-8") as stream:
        json.dump({"version": 1, "pairs": [[a, b, n] for (a, b), n in pairs.items()]}, stream,
                  ensure_ascii=False)


@lru_cache(maxsize=2)
def load_resources(dictionary, vocabulary, context, indexed, distance):
    spell = SymSpell(max_dictionary_edit_distance=distance, prefix_length=7)
    if not spell.load_dictionary(dictionary[0], 0, 1, encoding="utf-8"):
        raise ValueError("could not load frequency dictionary")
    with open(vocabulary[0], encoding="utf-8") as stream:
        words = frozenset(line.strip() for line in stream if line.strip())
    if not words:
        raise ValueError("vocabulary is empty")
    pairs = Counter()
    if context:
        if indexed:
            with open(context[0], encoding="utf-8") as stream:
                payload = json.load(stream)
            if payload["version"] != 1:
                raise ValueError("unsupported bigram index version")
            for a, b, n in payload["pairs"]:
                if not isinstance(n, int) or n <= 0:
                    raise ValueError("bigram counts must be positive integers")
                pairs[a, b] += n
        else:
            pairs = corpus_counts(context[0])
    totals = Counter()
    types = set()
    for (a, b), count in pairs.items():
        totals[a] += count
        types.update((a, b))
    # Include an unknown bucket; every conditional distribution uses the same support.
    return Resources(spell, words, dict(pairs), dict(totals), len(types | set(words)) + 1)
