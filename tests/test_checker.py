import tempfile
import unittest
from pathlib import Path
from dataclasses import replace
from toki_hebrew import HebrewChecker, CheckerConfig
from toki_hebrew.resources import write_bigram_index


class CheckerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root/'freq').write_text('אנחנו 100\nרוצים 90\nללמוד 80\nעברית 70\nספר 30\nסדר 30\nבספר 10\n',encoding='utf-8')
        (self.root/'vocab').write_text('אנחנו\nרוצים\nללמוד\nעברית\nספר\nסדר\nבספר\n',encoding='utf-8')
        (self.root/'corpus').write_text(('רוצים ספר ללמוד\n'*10)+'רוצים סדר עברית\n',encoding='utf-8')
        self.config=CheckerConfig(dictionary_path=self.root/'freq',vocabulary_path=self.root/'vocab',corpus_path=self.root/'corpus',min_score_margin=0)
        self.checker=HebrewChecker.load(self.config)

    def test_unicode_and_preservation(self):
        text='🙂  אנכנו, רוצים ללמוד עברית! hello אבabc שָׁלוֹם'
        result=self.checker.analyze(text)
        self.assertEqual(result.original_text,text)
        self.assertEqual(len(result.issues),1)
        issue=result.issues[0]
        self.assertEqual(text[issue.start:issue.end],'אנכנו')
        self.assertEqual(issue.utf16_start,issue.start+1)
        self.assertEqual(issue.suggestions[0].word,'אנחנו')
        self.assertEqual(issue.suggestions[0].edit_distance,1)

    def test_correct_and_empty(self):
        for text in ('', 'אנחנו רוצים ללמוד עברית', 'hello 🙂 123'):
            self.assertFalse(self.checker.analyze(text).issues)

    def test_cache_and_shared_resources(self):
        self.checker.analyze('אנכנו')
        self.checker.analyze('אנכנו')
        self.assertGreater(self.checker.generator.candidates.cache_info().hits,0)
        self.assertIs(HebrewChecker.load(self.config).resources,self.checker.resources)

    def test_abstention(self):
        checker=HebrewChecker.load(replace(self.config,min_score=1000))
        self.assertTrue(checker.analyze('אנכנו').issues[0].abstained)
        self.assertEqual(checker.analyze('קקקקקקק').issues[0].reason,'no_candidates')

    def test_context_and_index(self):
        result=self.checker.analyze('רוצים ספרר ללמוד')
        self.assertEqual(result.issues[0].suggestions[0].word,'ספר')
        index=self.root/'index.json'
        write_bigram_index(self.root/'corpus',index)
        other=HebrewChecker.load(replace(self.config,corpus_path=None,bigram_index_path=index))
        self.assertEqual(other.resources.bigrams,self.checker.resources.bigrams)

    def test_conditional_normalization_and_both_neighbors(self):
        from collections import Counter
        from toki_hebrew.resources import Resources
        from toki_hebrew.candidates import Candidate
        from toki_hebrew.ranking import rank
        pairs = {('רוצים','ספר'):10, ('רוצים','סדר'):10,
                 ('ספר','ללמוד'):1, ('סדר','ללמוד'):10}
        resources = Resources(None, frozenset({'ספר','סדר'}), pairs,
                              {'רוצים':20,'ספר':10000,'סדר':10}, 4)
        choices = [Candidate('ספר',1,30,False), Candidate('סדר',1,30,False)]
        self.assertEqual(rank(choices,'רוצים','ללמוד',resources,self.config)[0].word,'סדר')
        # No context means equal scores, even with very unequal conditional totals.
        ranked = rank(choices,None,None,resources,self.config)
        self.assertEqual(ranked[0].score,ranked[1].score)

    def test_prefix_surface_validation(self):
        options=self.checker.generator.candidates('בספרר')
        self.assertTrue(any(c.word=='בספר' for c in options))
        self.assertTrue(all(c.word in self.checker.resources.words for c in options))

    def test_punctuation_breaks_context(self):
        from unittest.mock import patch
        from toki_hebrew.ranking import rank
        with patch('toki_hebrew.api.rank', wraps=rank) as mocked:
            self.checker.analyze('רוצים, אנכנו! ללמוד')
            self.assertEqual(mocked.call_args.args[1:3],(None,None))

    def test_validation(self):
        for kw in ({'max_candidates':0},{'smoothing':0},{'ranker':'bad'},{'max_edit_distance':3}):
            with self.assertRaises(ValueError):
                CheckerConfig(**kw)

if __name__=='__main__':
    unittest.main()
