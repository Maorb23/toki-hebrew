# toki-hebrew

Lightweight, local, suggestion-only Hebrew spelling analysis for communication assistants. No LLM, network service, automatic correction, or message sending. Runtime dependency: SymSpellPy. Python 3.10+.

```sh
uv venv
uv pip install -e '.[dev]'
uv run pytest
# Also works without pytest, using the existing runtime environment:
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```

Provision your own licensed UTF-8 frequency dictionary (`word count` per line), attested surface vocabulary (one word per line), and optional independent corpus. Nothing downloads during import, loading, or unit tests.

```python
from toki_hebrew import HebrewChecker, CheckerConfig

checker = HebrewChecker.load(CheckerConfig(
    dictionary_path="/path/to/frequencies.txt",
    vocabulary_path="/path/to/vocabulary.txt",
    corpus_path="/path/to/independent-corpus.txt",
    max_edit_distance=2, max_candidates=40,
))
result = checker.analyze("אנכנו רוצים ללמוד עברית")
```

The short `HebrewChecker.load(CheckerConfig(...))` form uses `TOKI_HEBREW_DATA` (default `./data`) for `frequencies.txt` and `vocabulary.txt`. Missing files raise a filesystem error. Initialize once; reuse the checker. See `examples/integration.py` for JSON serialization.

`Analysis` contains untouched `original_text`, `issues`, and total/candidate/ranking milliseconds. Each issue has original text, ranked suggestions (word, edit distance, source frequency, prefix provenance, score), `abstained`, and a reason. Offsets are half-open: `start/end` count Python Unicode code points; `utf16_start/end` count UTF-16 code units for JavaScript. These are not byte or grapheme offsets. Scores are uncalibrated ranking values, not confidence probabilities. Abstention retains alternatives for review; a single candidate can still be wrong.

Spacing, punctuation, emoji, and foreign text remain in the original string. Mixed-script and pointed tokens are conservatively skipped. Context is limited to immediately adjacent Hebrew words separated by whitespace; punctuation and foreign words break context. Attested words are left alone: grammar and real-word errors are outside this checker's scope.

## Architecture and ranking

- `config.py`: validated, immutable paths, limits, weights and thresholds.
- `resources.py`: local vocabulary, SymSpell and bigram loading; bounded process resource cache keyed by resolved paths, size, mtime and edit radius. No inference disk access. Treat shared resources as read-only. Replacing a file and calling `load` refreshes resources; existing checkers retain their snapshot.
- `candidates.py`: SymSpell ALL lookup, validated single-prefix fallback, deterministic distance/frequency shortlist, bounded per-checker LRU cache.
- `ranking.py`: baseline raw bigram bonuses or additive-smoothed conditional log probabilities using `log P(candidate | previous) + log P(following | candidate)`. Unknown pairs receive finite probability; denominators account for conditioning-word frequency. Both use configurable distance, frequency and context weights.
- `api.py`: immutable suggestion records, Unicode spans, phase timing, optional abstention by absolute score or top-two margin. Defaults are heuristic, not tuned on the evaluation set. Set `min_score_margin=0` to disable margin abstention; `ranker="baseline", max_candidates=10, min_score_margin=0` reproduces prototype ranking.

Resource reuse avoids repeated initialization. Candidate caching avoids repeated edit searches; context ranking is always recomputed. `write_bigram_index(corpus, destination)` explicitly builds a versioned JSON count index; use `bigram_index_path` instead of `corpus_path` to avoid corpus tokenization at startup. JSON is portable, not necessarily smaller or faster; no speedup is claimed without measurement. SymSpell delete-index serialization is deferred because of portability/security and derived-data distribution concerns. Cache sizes bound candidate entries and resource snapshots, not total process RAM; deployment memory needs measurement.

## Evaluation

```sh
PYTHONPATH=src .venv/bin/python evaluation/benchmark.py
# optional local Dicta conversion; requires datasets in the preparation environment
.venv/bin/python scripts/prepare_vocabulary.py /path/to/wordlist-train.arrow data/vocabulary.txt
```

`evaluation/cases.json` preserves all 44 prototype examples, categorized as spelling, grammar, contextual, or correct. It is never loaded by runtime or corpus/index builders. These historical examples are a regression set, not evidence of unseen-data generalization; collect an independent Toki holdout before deployment or tuning. Corpus overlap with these examples has not been audited.

The benchmark reconstructs proposed sentences **only for evaluation**, uses aligned whitespace tokens for these substitution-only fixtures, and reports exact sentence accuracy, correction precision (correct replacements / all proposed replacements), recall (correct replacements / all gold erroneous tokens), false positives (changed correct tokens), and Recall@1/5/10/40 (gold candidate in returned list / all gold erroneous tokens, including undetected real-word/grammar errors). Abstention affects proposals but not candidate recall. Categories include correct contextual/grammar examples under `correct`; categories with errors identify intended capability. Baseline, conditional without abstention, and default conditional configurations are compared separately. First cache-cleared and repeated cached sentence passes report mean and nearest-rank p50/p95/p99 plus candidate and ranking phase times; they exclude initialization. Small samples and timer overhead limit percentile interpretation. Results and stage-level failure diagnoses are in `evaluation/results.json`; analysis is in `evaluation/REPORT.md`.

## Data and redistribution

The local Dicta attested wordlist is identified by the prototype as **CC BY-NC 4.0**. Commercial Toki use requires appropriate permission or a replacement vocabulary with suitable terms. Neither that vocabulary nor derived indexes are bundled by this package. `scripts/prepare_vocabulary.py` is an explicit offline converter for an already-cached Hugging Face `dicta-il/wordlist`, `attested` Arrow shard, retaining the actual Modern genre and observed prefix+wordform combinations. Its output inherits source restrictions. No license grant for that data is implied.

`he-100k.txt` came from the SymSpell frequency dictionary repository according to the legacy scripts; `hebrew_corpus.txt` was collected from `tomron87/hebrew-wikipedia-sentences-corpus` according to `download_heb_wik.py`. Their exact revisions, licenses, attribution and downstream distribution rights have not been independently verified. Keep these local and replaceable; obtain provenance and permissions before shipping data. The ignore rules exclude these files, DuckDB and local `data/`. Review any manually staged artifacts before the first commit. No software license has been selected on the owner's behalf.

Legacy prototype scripts remain for comparison, including their optional network-dependent workflows; do not import `test_dicta.py` in application code. Pytest discovery is restricted to `tests/`. No GitHub push or commit is performed.
