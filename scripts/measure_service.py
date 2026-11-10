"""Measure /ask without erasing failed attempts or reading gold labels.

No automatic retries. A 503/quota-like response stops further submissions;
in-flight requests still finish and are recorded. Concurrency is explicit,
default 1. Caller must establish account/quota permission before real runs.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import json
from pathlib import Path
import threading
import time
import urllib.error
import urllib.request


def measure(questions, base_url, out_dir, concurrency=1, pace_seconds=4):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Never overwrite an earlier experiment.
    attempts_file = (out_dir / 'attempts.jsonl').open('x')
    answers_file = (out_dir / 'answers.jsonl').open('x')
    stop = threading.Event()
    lock = threading.Lock()
    next_start = [0.0]
    records = []

    def request(q):
        with lock:
            delay = max(0, next_start[0] - time.monotonic())
            if delay:
                time.sleep(delay)
            if stop.is_set():
                return None, None
            next_start[0] = time.monotonic() + pace_seconds
        payload = {k: q[k] for k in ('id', 'question', 'role', 'as_of') if k in q}
        req = urllib.request.Request(base_url.rstrip('/') + '/ask',
                                     data=json.dumps(payload).encode(),
                                     headers={'Content-Type': 'application/json'})
        start = time.perf_counter()
        record = {'id': q['id'], 'attempt': 1, 'http_status': None,
                  'generator': None, 'error_category': None}
        answer = None
        try:
            with urllib.request.urlopen(req, timeout=120) as response:
                record['http_status'] = response.status
                record['generator'] = response.headers.get('X-Generator')
                obj = json.loads(response.read())
                answer = {k: obj[k] for k in ('id', 'status', 'answer', 'citations') if k in obj}
                if answer.get('id') != q['id'] or len(answer) != 4:
                    record['error_category'] = 'invalid_response'
                    answer = None
                    stop.set()
        except urllib.error.HTTPError as exc:
            record['http_status'] = exc.code
            record['retry_after'] = exc.headers.get('Retry-After')
            record['error_category'] = 'http_error'
            # Do not retain provider content or infer a 503 cause.
            if exc.code in (429, 503):
                stop.set()
        except Exception:
            record['error_category'] = 'transport_or_parse_error'
            stop.set()
        record['elapsed_ms'] = round((time.perf_counter() - start) * 1000, 3)
        return record, answer

    iterator = iter(questions)
    with attempts_file, answers_file, ThreadPoolExecutor(max_workers=concurrency) as pool:
        pending = set()
        for _ in range(concurrency):
            q = next(iterator, None)
            if q is not None:
                pending.add(pool.submit(request, q))
        while pending:
            completed, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in completed:
                record, answer = future.result()
                if record is not None:
                    records.append(record)
                    attempts_file.write(json.dumps(record) + '\n')
                    attempts_file.flush()
                    if answer is not None:
                        answers_file.write(json.dumps(answer) + '\n')
                        answers_file.flush()
                if not stop.is_set():
                    q = next(iterator, None)
                    if q is not None:
                        pending.add(pool.submit(request, q))
    summary = {'requested_questions': len(questions), 'attempted': len(records),
               'successful': sum(r['http_status'] == 200 and r['error_category'] is None for r in records),
               'stopped_early': stop.is_set(), 'concurrency': concurrency,
               'pace_seconds': pace_seconds, 'automatic_client_retries': 0,
               'latency_scope': 'every attempted HTTP request, including failures; no recovery waits',
               'tokens_and_cost': 'require independent per-request service meter; not inferred from answers'}
    (out_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--questions', required=True)
    ap.add_argument('--base-url', default='http://127.0.0.1:8123')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--concurrency', type=int, default=1)
    ap.add_argument('--pace-seconds', type=float, default=4)
    args = ap.parse_args()
    if args.concurrency < 1 or args.pace_seconds < 0:
        ap.error('concurrency must be positive and pacing nonnegative')
    questions = [json.loads(line) for line in Path(args.questions).read_text().splitlines() if line.strip()]
    print(json.dumps(measure(questions, args.base_url, args.out_dir,
                             args.concurrency, args.pace_seconds), indent=2))


if __name__ == '__main__':
    main()
