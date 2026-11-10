import json, time, urllib.request
qs = {json.loads(l)['id']: json.loads(l) for l in open('/home/sandbox/op05/data/questions_dev.jsonl')}
have = {json.loads(l)['id'] for l in open('/home/sandbox/op05/results/raw/answers_dev_llm_v4A_2026-11-10.jsonl')}
missing = [q for i, q in qs.items() if i not in have]
print("missing:", len(missing))
extra = []
for q in missing:
    body = json.dumps({"id": q['id'], "question": q['question'], "role": q['role'],
                       "as_of": q.get('as_of','9999-12-31')}).encode()
    req = urllib.request.Request("http://127.0.0.1:8123/ask", data=body, headers={"Content-Type":"application/json"})
    for attempt in (1, 2, 3):
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                resp = json.loads(r.read())
            ms = (time.perf_counter()-t0)*1000
            row = {"id": q['id'], "status": resp.get('status'), "answer": resp.get('answer',''), "citations": resp.get('citations',[])}
            extra.append((q['id'], row, round(ms,1)))
            print(f"OK {q['id']} {ms:.0f}ms {resp.get('status')} (attempt {attempt})", flush=True)
            break
        except Exception as e:
            print(f"FAIL {q['id']} attempt {attempt}: {e}", flush=True)
            time.sleep(15)
    time.sleep(6)
with open('/home/sandbox/op05/results/raw/answers_dev_llm_v4A_2026-11-10.jsonl','a') as f:
    for _, row, _ in extra:
        f.write(json.dumps(row)+"\n")
with open('/tmp/redrive_lat.json','w') as f:
    json.dump([{"id": i, "ms": m} for i, _, m in extra], f)
print("REDRIVE DONE", len(extra), "of", len(missing), flush=True)
