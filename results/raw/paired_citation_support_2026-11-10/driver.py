"""Bounded paired support comparison: fixed answers, same rubric/model.

Original/derived arms alternate order per row. No retries on any transport
error, invalid verdict stops. Uses the established Judge payload/parsing,
with a strict transport wrapper to prevent its ordinary error retries.
"""
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import time
import urllib.request

ROOT = Path('/home/sandbox/op05')
spec = importlib.util.spec_from_file_location('op05_paired_grader', ROOT/'tools/op05_grader/grader.py')
g = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g)
original = {r['id']:r for r in g.load_jsonl(str(ROOT/'results/raw/answers_dev_llm_v4A_2026-11-10.jsonl'))}
derived = {r['id']:r for r in g.load_jsonl(str(ROOT/'results/raw/answers_v4A_offline_citation_spans_UNJUDGED_2026-11-10.jsonl'))}
questions = {r['id']:r for r in g.load_jsonl(str(ROOT/'data/questions_dev.jsonl'))}
ids = [i for i in original if original[i]['citations'] != derived[i]['citations']]
assert len(ids) == 13
assert all(original[i]['answer'] == derived[i]['answer'] for i in ids)
BASE = ROOT/'results/raw/paired_citation_support_2026-11-10'
BASE.mkdir(exist_ok=False)
(ROOT/'results/raw/paired_citation_support_2026-11-10/method.json').write_text(json.dumps({
    'model':'gemini-3.5-flash-lite', 'rubric':'unchanged B', 'rows':ids,
    'sample':'13 changed rows selected offline, not random',
    'order':'alternate arm order per row', 'retries':0,
    'key':'positively mapped free-tier fallback project only, no key failover',
    'answer_text':'identical per pair', 'outcome':'support only; no correctness/ready claim'},indent=2)+'\n')
os.environ['LLM_API_KEY'] = Path('/home/sandbox/.vault/gemini-api-key-op-05-fallback-project-op-05-eval-2-8b9e3d/password').read_text().strip()
os.environ.pop('LLM_API_KEY_FALLBACK',None)
os.environ['JUDGE_DELAY_SEC'] = '5'
real_open = urllib.request.urlopen
usage = {}

def strict_open(*args, **kwargs):
    global usage
    try:
        response = real_open(*args, **kwargs)
        with response:
            body = response.read()
        data = json.loads(body)
        usage = data.get('usage') or {}
        return io.BytesIO(body)
    except BaseException as exc:
        # No provider response body or credential is printed.
        raise SystemExit('STOP transport signal: ' + type(exc).__name__ +
                         (' HTTP ' + str(exc.code) if hasattr(exc,'code') else ''))

g.urllib.request.urlopen = strict_open
judge = g.Judge('https://generativelanguage.googleapis.com/v1beta/openai/', 'gemini-3.5-flash-lite', True)
with (BASE/'verdicts.jsonl').open('x') as out:
    for ix, qid in enumerate(ids):
        arms = ['original','derived'] if ix % 2 == 0 else ['derived','original']
        for arm in arms:
            row = original[qid] if arm == 'original' else derived[qid]
            for ci,c in enumerate(row['citations']):
                prompt = g._prompt('B').format(question=questions[qid]['question'],
                    candidate_answer=row['answer'],doc_id=c['doc_id'],section=c['section'],quote_with_context=c['quote'])
                t0=time.perf_counter()
                verdict=judge.ask(prompt,('SUPPORTS','DOES_NOT_SUPPORT'))
                record={'id':qid,'arm':arm,'citation_index':ci,'verdict':verdict,
                    'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),
                    'usage':usage,'elapsed_ms':round((time.perf_counter()-t0)*1000,1)}
                out.write(json.dumps(record)+'\n');out.flush()
                print(qid,arm,ci,verdict,flush=True)
                if verdict not in ('SUPPORTS','DOES_NOT_SUPPORT'):
                    raise SystemExit('STOP invalid verdict')
print('COMPLETE',judge.calls,flush=True)
