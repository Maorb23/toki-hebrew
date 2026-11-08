
import importlib.util
import time
from huggingface_hub import hf_hub_download

path = hf_hub_download(
    repo_id="dicta-il/wordlist",
    filename="wordlist_query.py",
    repo_type="dataset",
)

spec = importlib.util.spec_from_file_location("wq", path)
wq = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wq)

start = time.perf_counter()
lexicon = wq.WordlistQuery(genre="Modern", disk=True)
print(f"Initialization: {time.perf_counter()-start:.2f}s")

words = [
    "הילדים",
    "ילדים",
    "במשרד",
    "ללמוד",
    "תפוח",
    "אנחנו",
    "הולכות",
    "הילידם",
    "אנכנו",
]

for word in words:
    matches = lexicon.get_word_options(word)

    print(f"\nWORD: {word}")
    print(f"Total readings: {len(matches)}")

    for match in matches[:5]:
        print(match)

lexicon.close()

