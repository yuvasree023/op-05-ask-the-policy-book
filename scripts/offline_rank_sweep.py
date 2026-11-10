import json
from collections import defaultdict
from pathlib import Path
from policybook.answer import Answerer
from policybook.llm import pack_evidence
qs=[json.loads(l) for l in open('data/questions_dev.jsonl')]
a=Answerer('.index','models/all-MiniLM-L6-v2','data/corpus')
weights={'rrf_equal':(1,1),'rrf_lexical_2x':(1,2),'rrf_lexical_3x':(1,3),'lexical_only':(0,1),'vector_only':(1,0)}
stats={n:defaultdict(int) for n in weights}; changes={n:[] for n in weights}
for q in qs:
 if q['expected_status']!='answered':continue
 r=a.retr.retrieve(q['question'],q['role'],q['as_of'],k=1000)
 gold={c['section'] for c in q.get('gold_citations',[])}
 base=None
 for n,(wv,wl) in weights.items():
  hits=sorted(r.hits,key=lambda h:(-(wv/(60+h.vec_rank)+wl/(60+h.bm25_rank)),h.chunk_id))
  packed=pack_evidence(hits[:8],a._gen_meta(q['as_of']))
  present={s for s in gold if any(s in b for b in packed)}
  stats[n]['n']+=1
  for k in [1,3,6,8,20]:stats[n]['any_gold_top'+str(k)]+=bool(gold & {h.section for h in hits[:k]})
  stats[n]['any_gold_in_pack']+=bool(present)
  stats[n]['all_gold_in_pack']+=bool(gold and present==gold)
  stats[n]['pack_chars_total']+=len('\n\n'.join(packed))
  if n=='rrf_equal':base=bool(present)
  elif bool(present)!=base:changes[n].append({'id':q['id'],'pack_gold_present':bool(present)})
report={'scope':'published dev only; offline ranking/pack ablation, no model calls; refusals and role signals NOT re-measured; no runtime change','weights':weights,'counts':stats,'per_question_pack_presence_changes':changes}
Path('results/raw/offline_rank_sweep_2026-11-10.json').write_text(json.dumps(report,indent=2)+'\n')
for n,s in stats.items():print(n,dict(s),'packchanges',len(changes[n]))
