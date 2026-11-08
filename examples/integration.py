"""Local suggestion payload for a communication assistant; no sending or editing."""
from dataclasses import asdict
import json
from toki_hebrew import HebrewChecker, CheckerConfig

# Provision licensed local files in TOKI_HEBREW_DATA (default: ./data).
# Construct once at application startup and reuse for subsequent messages.
checker = HebrewChecker.load(CheckerConfig(max_edit_distance=2, max_candidates=40))
result = checker.analyze("אנכנו רוצים ללמוד עברית 🙂")
print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
