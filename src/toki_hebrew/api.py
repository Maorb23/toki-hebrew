from dataclasses import dataclass
import re
from time import perf_counter
from .config import CheckerConfig
from .resources import TOKEN, load_resources, signature
from .candidates import CandidateGenerator
from .ranking import Suggestion, rank


@dataclass(frozen=True)
class Issue:
    original: str
    start: int
    end: int
    utf16_start: int
    utf16_end: int
    suggestions: tuple[Suggestion, ...]
    abstained: bool
    reason: str


@dataclass(frozen=True)
class Analysis:
    original_text: str
    issues: tuple[Issue, ...]
    latency_ms: float
    candidate_ms: float
    ranking_ms: float


class HebrewChecker:
    def __init__(self, resources, config):
        self.resources, self.config = resources, config
        self.generator = CandidateGenerator(resources, config)
        self.initialization_ms = 0.0

    @classmethod
    def load(cls, config=None):
        start = perf_counter()
        config = config or CheckerConfig()
        dictionary, vocabulary = config.paths()
        context = config.corpus_path or config.bigram_index_path
        resources = load_resources(signature(dictionary), signature(vocabulary),
                                   signature(context) if context else None,
                                   bool(config.bigram_index_path), config.max_edit_distance)
        checker = cls(resources, config)
        checker.initialization_ms = (perf_counter() - start) * 1000
        return checker

    def analyze(self, text: str) -> Analysis:
        if not isinstance(text, str):
            raise TypeError("text must be str")
        start = perf_counter()
        matches = list(TOKEN.finditer(text))
        issues = []
        generation = ranking = 0.0
        # Prefix table keeps UTF-16 conversion linear even for long messages.
        utf16 = [0]
        for char in text:
            utf16.append(utf16[-1] + (2 if ord(char) > 0xFFFF else 1))
        for i, match in enumerate(matches):
            word = match.group()
            # Mixed scripts and niqqud are conservatively left untouched.
            if not re.fullmatch(r"[א-ת]+", word) or word in self.resources.words:
                continue
            def neighbor(j):
                if not 0 <= j < len(matches):
                    return None
                other = matches[j]
                gap = text[other.end():match.start()] if j < i else text[match.end():other.start()]
                return other.group() if gap.isspace() and re.fullmatch(r"[א-ת]+", other.group()) else None
            tick = perf_counter()
            choices = self.generator.candidates(word)
            generation += perf_counter() - tick
            tick = perf_counter()
            suggestions = rank(choices, neighbor(i - 1), neighbor(i + 1), self.resources, self.config)
            ranking += perf_counter() - tick
            reason = "suggested"
            if not suggestions:
                reason = "no_candidates"
            elif self.config.min_score is not None and suggestions[0].score < self.config.min_score:
                reason = "low_score"
            elif len(suggestions) > 1 and suggestions[0].score - suggestions[1].score < self.config.min_score_margin:
                reason = "small_margin"
            issues.append(Issue(word, match.start(), match.end(), utf16[match.start()], utf16[match.end()],
                                suggestions, reason != "suggested", reason))
        return Analysis(text, tuple(issues), (perf_counter() - start) * 1000, generation * 1000, ranking * 1000)
