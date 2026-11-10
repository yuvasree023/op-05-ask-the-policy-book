import json, time, urllib.request, sys
qs = [json.loads(l) for l in open('/home/sandbox/op05/data/questions_dev.jsonl')]
out = open('/home/sandbox/op05/results/raw/answers_dev_llm_v4A_2026-11-10.jsonl','w')
lat = open('/home/sandbox/op05/results/raw/service_v4A_2026-11-10_latencies.json','w')
lat.write('{"workload":"single-client sequential, 4s pacing, retry flag ON, real model-backed","requests":[')
first = True
for i, q in enumerate(qs):
    body = json.dumps({"id": q['id'], "question": q['question'], "role": q['role'],
                       "as_of": q.get('as_of','9999-12-31')}).encode()
    req = urllib.request.Request("http://127.0.0.1:8123/ask", data=body,
                                 headers={"Content-Type":"application/json"})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            resp = json.loads(r.read())
        ms = (time.perf_counter()-t0)*1000
        resp_row = {"id": q['id'], "status": resp.get('status'), "answer": resp.get('answer',''),
                    "citations": resp.get('citations',[])}
        out.write(json.dumps(resp_row)+"\n"); out.flush()
        lat.write(("" if first else ",") + json.dumps({"id": q['id'], "ms": round(ms,1), "status": resp.get('status'), "generator": resp.get('generator')}))
        first = False
        print(f"{i+1}/146 {q['id']} {ms:.0f}ms {resp.get('status')}", flush=True)
    except Exception as e:
        print(f"{i+1}/146 {q['id']} ERROR {e}", flush=True)
    time.sleep(4)
lat.write("]}")
lat.close(); out.close()
print("DRIVER DONE", flush=True)
