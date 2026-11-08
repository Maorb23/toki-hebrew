from dataclasses import dataclass
import math
from .candidates import Candidate


@dataclass(frozen=True)
class Suggestion(Candidate):
    score: float


def rank(candidates, previous, following, resources, config):
    def score(c):
        value = -config.distance_weight * c.edit_distance + config.frequency_weight * math.log1p(c.frequency)
        if resources.bigrams:
            for left, right in ((previous, c.word), (c.word, following)):
                if left is None or right is None:
                    continue
                count = resources.bigrams.get((left, right), 0)
                if config.ranker == "baseline":
                    value += config.context_weight * math.log1p(count)
                else:
                    # Markov factorization: log P(candidate|left) + log P(right|candidate).
                    # Unlike raw bonuses, normalize by the conditioning word's total.
                    value += config.context_weight * math.log(
                        (count + config.smoothing) /
                        (resources.totals.get(left, 0) + config.smoothing * resources.vocabulary_size))
        return value
    suggestions = [Suggestion(c.word, c.edit_distance, c.frequency, c.via_prefix, score(c)) for c in candidates]
    return tuple(sorted(suggestions, key=lambda c: (-c.score, c.edit_distance, c.word)))
