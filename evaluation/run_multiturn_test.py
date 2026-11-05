"""Live two-turn clarification test (uses the real model).

Turn 1 is ambiguous (two Google applications match). Turn 2 sends only an id. The
agent must continue the original goal from turn 1 instead of starting a new task.

  python -m evaluation.run_multiturn_test                      # in-process app
  python -m evaluation.run_multiturn_test --url https://YOUR-APP.onrender.com
"""
import argparse
import json
import time
import uuid

from evaluation.run_public_tests import RESULTS

TURNS = ['Move my Google software engineer application to interviewed',
         'app-001']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='')
    parser.add_argument('--label', default='local')
    args = parser.parse_args()
    session = 'multiturn-' + uuid.uuid4().hex[:16]

    if args.url:
        import httpx
        client = httpx.Client(base_url=args.url.rstrip('/'), timeout=70)
        ctx = None
    else:
        from fastapi.testclient import TestClient
        from app.main import app
        ctx = TestClient(app)
        client = ctx.__enter__()

    transcript = []
    try:
        for task in TURNS:
            r = client.post('/chat', json={'session_id': session, 'task': task}).json()
            transcript.append({'user': task, 'status': r['status'], 'stop_reason': r['stop_reason'],
                               'agent': r['final_response'], 'pending_goal': r.get('pending_goal'),
                               'tools': [(c['tool'], c['outcome']) for c in r['tool_calls']]})
            print(f"USER:  {task}\nAGENT: [{r['status']}] {r['final_response']}\n")
        board = client.get(f'/chat/{session}/board').json()['records']
        final = next(x for x in board if x['id'] == 'app-001')['status']
    finally:
        if ctx:
            ctx.__exit__(None, None, None)

    passed = (transcript[0]['status'] == 'needs_clarification'
              and transcript[1]['status'] == 'completed'
              and ('update_status', 'success') in [tuple(t) for t in transcript[1]['tools']]
              and final == 'interviewed')
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f'multiturn_{args.label}.json'
    out.write_text(json.dumps({'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                               'passed': passed, 'app_001_final_status': final,
                               'transcript': transcript}, indent=2), encoding='utf-8')
    print(('PASS' if passed else 'FAIL') + f' multi-turn clarification -> {out.name}')


if __name__ == '__main__':
    main()
