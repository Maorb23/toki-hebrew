from symspellpy import SymSpell, Verbosity

spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
spell.load_dictionary("he-100k.txt", 0, 1)

words = ["הילדים", "הילידם", "אנכנו", "ללמוד", "במשרד", "תפו"]

for word in words:
    suggestions = spell.lookup(
        word,
        Verbosity.ALL,
        max_edit_distance=2,
        include_unknown=True,
    )
    print(word, [(s.term, s.distance, s.count) for s in suggestions[:5]])
