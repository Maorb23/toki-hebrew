from dataclasses import dataclass
from functools import lru_cache
from symspellpy import Verbosity


@dataclass(frozen=True)
class Candidate:
    word: str
    edit_distance: int
    frequency: int
    via_prefix: bool


class CandidateGenerator:
    def __init__(self, resources, config):
        self.resources, self.config = resources, config
        self.candidates = lru_cache(maxsize=config.cache_size)(self._generate)

    def _generate(self, word):
        found = {}

        def offer(term, suggestion, prefix):
            if term == word or term not in self.resources.words:
                return
            candidate = Candidate(term, suggestion.distance, suggestion.count, prefix)
            old = found.get(term)
            if old is None or (candidate.edit_distance, -candidate.frequency) < (old.edit_distance, -old.frequency):
                found[term] = candidate

        for s in self.resources.spell.lookup(word, Verbosity.ALL, max_edit_distance=self.config.max_edit_distance):
            offer(s.term, s, False)
        # One retained prefix is a search fallback, not a morphological validity rule.
        # Validate the complete surface form against the attested vocabulary.
        if len(word) >= 4 and word[0] in "והבכלמש":
            for s in self.resources.spell.lookup(word[1:], Verbosity.ALL, max_edit_distance=self.config.max_edit_distance):
                offer(word[0] + s.term, s, True)
        return tuple(sorted(found.values(), key=lambda c: (c.edit_distance, -c.frequency, c.word))[:self.config.max_candidates])
