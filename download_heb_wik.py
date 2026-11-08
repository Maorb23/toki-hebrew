
from datasets import load_dataset
from itertools import islice
from pathlib import Path
import time

# 1. Stream sentences without downloading the entire dataset.
dataset = load_dataset(
    "tomron87/hebrew-wikipedia-sentences-corpus",
    split="train",
    streaming=True,
)

# 2. Start with 100,000 sentences.
limit = 100_000
output = Path("hebrew_corpus.txt")

start = time.perf_counter()

# 3. Write one sentence per line.
with output.open("w", encoding="utf-8") as f:
    for row in islice(dataset, limit):
        sentence = row["sentence"].strip()
        if sentence:
            f.write(sentence + "\n")

elapsed = time.perf_counter() - start

print(f"Corpus saved: {output}")
print(f"Time: {elapsed:.2f}s")
print(f"Size: {output.stat().st_size / 1024**2:.2f} MB")

