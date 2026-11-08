"""Explicit local-only extraction from a cached HF attested Arrow shard.
Requires datasets in the preparation environment; no network or remote code.
"""
import argparse
from pathlib import Path
from datasets import Dataset

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("arrow", type=Path)
p.add_argument("output", type=Path)
a = p.parse_args()
ds = Dataset.from_file(str(a.arrow))
modern = ds.features["genres"].feature.str2int("Modern")
words = {(row["prefix"] or "") + row["wordform"] for row in ds if modern in row["genres"]}
a.output.parent.mkdir(parents=True, exist_ok=True)
a.output.write_text("\n".join(sorted(words)) + "\n", encoding="utf-8")
print(f"Wrote {len(words)} restricted wordforms; do not redistribute without permission")
