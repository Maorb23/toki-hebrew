from dataclasses import dataclass
from pathlib import Path
import math
import os


@dataclass(frozen=True)
class CheckerConfig:
    dictionary_path: str | Path | None = None
    vocabulary_path: str | Path | None = None
    corpus_path: str | Path | None = None
    bigram_index_path: str | Path | None = None
    max_edit_distance: int = 2
    max_candidates: int = 40
    cache_size: int = 4096
    ranker: str = "conditional"
    distance_weight: float = 2.0
    frequency_weight: float = 0.25
    context_weight: float = 1.5
    smoothing: float = 0.1
    min_score_margin: float = 0.5
    min_score: float | None = None

    def __post_init__(self):
        if self.max_edit_distance not in (1, 2) or self.max_candidates < 1 or self.cache_size < 0:
            raise ValueError("distance must be 1 or 2; candidates positive; cache nonnegative")
        if self.ranker not in ("baseline", "conditional"):
            raise ValueError("ranker must be baseline or conditional")
        values = (self.distance_weight, self.frequency_weight, self.context_weight,
                  self.smoothing, self.min_score_margin)
        if any(not math.isfinite(x) or x < 0 for x in values) or self.smoothing == 0:
            raise ValueError("weights/margin must be finite and nonnegative; smoothing positive")
        if self.min_score is not None and not math.isfinite(self.min_score):
            raise ValueError("min_score must be finite")
        if self.corpus_path and self.bigram_index_path:
            raise ValueError("choose corpus_path or bigram_index_path")

    def paths(self):
        # No implicit downloads or dependence on the caller's working directory.
        root = Path(os.environ.get("TOKI_HEBREW_DATA", "data"))
        return (Path(self.dictionary_path or root / "frequencies.txt").resolve(),
                Path(self.vocabulary_path or root / "vocabulary.txt").resolve())
