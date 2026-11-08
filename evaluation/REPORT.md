# Local regression report

Measured on 2026-11-08, macOS arm64, Python 3.13.18, SymSpellPy 6.10.2. Local cached Dicta Modern extraction: 215,739 surface forms. Frequency and Wikipedia corpus files are the existing repository inputs, not downloaded or modified. Each configuration starts with the shared resource cache cleared; filesystem pages may already be cached. One 44-sentence pass per configuration, followed immediately per sentence by a repeated candidate-cached analysis. These are small-sample diagnostic timings, not a load test.

| Metric | Baseline (10 candidates) | Conditional (40, no abstention) | Conditional (40, margin 0.5 default) |
|---|---:|---:|---:|
| Exact sentences | 29/44 | 29/44 | 27/44 |
| Correction precision | 62.5% | 62.5% | 100% |
| Correction recall | 38.46% | 38.46% | 26.92% |
| False positive changes to correct tokens | 0 | 0 | 0 |
| Recall@1 | 38.46% | 38.46% | 38.46% |
| Recall@5 / @10 | 50% / 50% | 50% / 50% | 50% / 50% |
| Recall@40 (returned list) | 50% | 57.69% | 57.69% |
| Initialization ms | 1094.48 | 1247.75 | 1171.94 |
| Uncached mean ms | 1.529 | 1.572 | 1.521 |
| Uncached p50 / p95 / p99 ms | .0035 / 8.184 / 10.536 | .0038 / 8.519 / 10.359 | .0033 / 8.224 / 10.020 |
| Cached mean ms | .0066 | .0174 | .0132 |
| Cached p50 / p95 / p99 ms | .0032 / .0212 / .0250 | .0032 / .0413 / .1869 | .0032 / .0344 / .0892 |
| Candidate generation mean ms | 1.520 | 1.544 | 1.498 |
| Ranking mean ms | .0054 | .0227 | .0185 |

The baseline reproduces the reported 29/44 accuracy. The historical ~1.25 ms warm average did not distinguish cache states, so it is not directly comparable to either new timing column. The refactor's cache-cleared baseline averages 1.53 ms and cached baseline 0.0066 ms. Conditional normalization is mathematically better-founded, but yields no exact accuracy gain here; abstention reduces incorrect suggestions at the cost of missed correct suggestions. Do not tune thresholds on these historical evaluation examples.

All three configurations preserve all 23 correct-text examples. They miss the five grammar and one contextual real-word error. Baseline and conditional without abstention solve 6/15 erroneous spelling sentences; default abstention solves 4/15. Token metrics include 26 gold error positions across categories, rather than treating wrong replacements as successful detection. False positives count changes to gold-correct tokens, while wrong replacements at error positions reduce correction precision.

## Named failures, without hardcoded correction rules

An additional diagnostic enumerated the full validated candidate list before truncation using the same distance/frequency ordering:

| Input → target | Finding |
|---|---|
| תפו → תפוח | Target is valid and present in the frequency dictionary; shortlist position 11, excluded by baseline's top 10. Included at 40 but loses ranking. Default abstains. |
| היופ → היום | Target is valid and shortlist position 2. Ranking favors היו; default abstains. |
| נפיגש → ניפגש | Target is valid and shortlist position 3. Ranking favors ניגש; default abstains. |
| מתכבן → מתכוון | Target is valid and shortlist position 33. Baseline excludes it; top 40 includes it but ranking favors מתכנן. Default abstains. |
| נימצא → נמצא | Both are accepted by the attested vocabulary. Input bypasses correction entirely: vocabulary policy, not candidate generation or ranking. |

Attestation is not a normative spelling standard. Improving the last case requires a replaceable normative lexicon or a separately evaluated real-word detection policy; it cannot be solved by ranking candidates for unknown words. The generator's shortlist optimizes speed, but may discard contextually appropriate forms before ranking. Increasing its size improves candidate recall here without improving top-one accuracy. The weak corpus may lack useful contexts, and original neighbors may themselves be erroneous. No failure-specific replacement rules were added.

## Verification and limits

Nine offline unit tests pass, covering unchanged input, emoji/UTF-16 offsets, punctuation context boundaries, pointed/mixed text, candidate caching, resource sharing, prefix validation, conditional denominators and both neighbors, index roundtrip, empty input, config validation and abstention. The full categorized benchmark ran against actual local resources; aggregate results and per-error stage/gold-rank diagnostics are in `results.json`.

`uv pip install -e '.[dev]'` could not complete: the default cache directory was not writable, and retrying with `/tmp/toki-uv` failed DNS access to PyPI. Therefore pytest invocation, wheel building, clean-environment installation, Python 3.10 compatibility, concurrent load behavior, memory usage and index startup speed are not verified. Tests ran with the existing `.venv`, `PYTHONPATH=src`, and standard-library unittest. No large data downloads occurred. Data license and corpus provenance statements come from the existing prototype and have not been independently audited. No git repository existed initially; no commit or remote push was made.
