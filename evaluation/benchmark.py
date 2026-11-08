"""Run from repo root: PYTHONPATH=src python evaluation/benchmark.py."""
import argparse
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
import statistics
from toki_hebrew import CheckerConfig, HebrewChecker


def percentiles(values):
    values = sorted(values)
    return {"mean": statistics.mean(values), **{f"p{p}": values[max(0, __import__("math").ceil(len(values)*p/100)-1)] for p in (50,95,99)}}


def evaluate(checker, cases):
    counts = Counter()
    uncached, cached, generation, ranking = [], [], [], []
    categories = {}
    failures = []
    for case in cases:
        checker.generator.candidates.cache_clear()
        result = checker.analyze(case['text'])
        uncached.append(result.latency_ms)
        generation.append(result.candidate_ms)
        ranking.append(result.ranking_ms)
        # Evaluation-only reconstruction: the public API never applies changes.
        tokens = case['text'].split()
        gold = case['expected'].split()
        assert len(tokens) == len(gold), 'This fixture uses aligned word substitutions only'
        issue_by_start = {i.start:i for i in result.issues}
        offset = 0
        predicted = []
        for source, target in zip(tokens, gold):
            issue = issue_by_start.get(offset)
            proposal = issue.suggestions[0].word if issue and issue.suggestions and not issue.abstained else source
            predicted.append(proposal)
            error = source != target
            changed = proposal != source
            counts['errors'] += error
            counts['proposals'] += changed
            counts['tp'] += error and proposal == target
            counts['false_positives'] += not error and changed
            if error:
                options = issue.suggestions if issue else ()
                for k in (1,5,10,40):
                    counts[f'recall@{k}'] += any(s.word == target for s in options[:k])
                if proposal != target:
                    all_candidates = checker.generator.candidates(source)
                    stage = ('vocabulary_target' if target not in checker.resources.words else
                             'vocabulary_accepts_input' if source in checker.resources.words else
                             'candidate_generation' if not any(c.word == target for c in all_candidates) else
                             'abstention' if issue.abstained else 'ranking')
                    failures.append(dict(source=source,target=target,stage=stage,
                                         gold_rank=next((j+1 for j,s in enumerate(options) if s.word == target), None)))
            offset += len(source)+1
        exact = ' '.join(predicted) == case['expected']
        counts['exact'] += exact
        bucket = categories.setdefault(case['category'],[0,0])
        bucket[0] += exact
        bucket[1] += 1
        cached.append(checker.analyze(case['text']).latency_ms)
    return dict(exact=f"{counts['exact']}/{len(cases)}", categories=categories,
                precision=counts['tp']/counts['proposals'] if counts['proposals'] else None,
                recall=counts['tp']/counts['errors'], false_positives=counts['false_positives'],
                recall_at_k={str(k):counts[f'recall@{k}']/counts['errors'] for k in (1,5,10,40)},
                initialization_ms=checker.initialization_ms, uncached_ms=percentiles(uncached),
                cached_ms=percentiles(cached), generation_ms=percentiles(generation),
                ranking_ms=percentiles(ranking), failures=failures)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dictionary',default='he-100k.txt')
    p.add_argument('--vocabulary',default='data/vocabulary.txt')
    p.add_argument('--corpus',default='hebrew_corpus.txt')
    p.add_argument('--output',default='evaluation/results.json')
    args=p.parse_args()
    cases=json.loads(Path('evaluation/cases.json').read_text())
    config=CheckerConfig(dictionary_path=args.dictionary,vocabulary_path=args.vocabulary,corpus_path=args.corpus)
    reports={}
    for name,cfg in [('baseline',replace(config,ranker='baseline',max_candidates=10,min_score_margin=0)),
                     ('conditional_no_abstention',replace(config,min_score_margin=0)),
                     ('conditional',config)]:
        # Include full resource initialization for each ranker, rather than shared-cache time.
        from toki_hebrew.resources import load_resources
        load_resources.cache_clear()
        reports[name]=evaluate(HebrewChecker.load(cfg),cases)
    Path(args.output).write_text(json.dumps(reports,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:{a:b for a,b in v.items() if a!='failures'} for k,v in reports.items()},indent=2))

if __name__=='__main__':
    main()
