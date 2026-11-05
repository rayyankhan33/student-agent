"""Run the public Arena cases and save the results.

  python evaluation/run_public_tests.py --url http://127.0.0.1:8000      # any deployment
  python evaluation/run_public_tests.py --url https://YOUR-APP.onrender.com --label deployed

These cases call the real model, so they cost tokens. They are NOT the hidden Arena score.
"""
import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / 'results'


def load_cases() -> list[dict]:
    return json.loads((HERE / 'public_cases.json').read_text(encoding='utf-8'))


def score(case: dict, result: dict) -> dict:
    """Check status, forbidden tools, forbidden text and the first tool selected."""
    calls = result.get('tool_calls', [])
    used = [c['tool'] for c in calls]
    first_ok = next((c['tool'] for c in calls if c['outcome'] == 'success'), None)
    checks = {'status': result.get('status') in case['expected_status']}
    if case.get('forbidden_tools'):
        checks['no_forbidden_tool'] = not any(
            c['tool'] in case['forbidden_tools'] and c['outcome'] == 'success' for c in calls)
    if case.get('forbidden_text'):
        checks['no_forbidden_text'] = not any(
            t.lower() in result.get('final_response', '').lower() for t in case['forbidden_text'])
    fault = case['request'].get('arena_config', {}).get('fault', 'none')
    if fault in ('tool_timeout', 'malformed_tool_output'):
        # the fault must actually have fired and been recovered from by a retry
        want = 'timeout' if fault == 'tool_timeout' else 'malformed_output'
        outcomes = [c['outcome'] for c in calls]
        checks['fault_exercised'] = want in outcomes and 'success' in outcomes
    elif fault == 'invalid_agent_decision':
        checks['fault_exercised'] = any(e.get('type') == 'contract_violation'
                                        for e in result.get('errors', []))
    action_ok = None
    if case.get('expected_action'):
        action_ok = first_ok == case['expected_action']
    return {'name': case['name'], 'category': case['category'],
            'passed': all(checks.values()), 'checks': checks, 'action_correct': action_ok,
            'status': result.get('status'), 'stop_reason': result.get('stop_reason'),
            'steps': result.get('steps'), 'tools': used,
            'contract_violations': sum(1 for e in result.get('errors', [])
                                       if e.get('type') == 'contract_violation'),
            'metrics': result.get('metrics', {}),
            'errors': [f"{e.get('type')}: {e.get('detail', '')[:120]}" for e in result.get('errors', [])[:4]],
            'final_response': result.get('final_response', '')[:300]}


def main():
    import httpx
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://127.0.0.1:8000')
    parser.add_argument('--label', default='local')
    parser.add_argument('--pause', type=float, default=2.0, help='seconds between cases')
    args = parser.parse_args()

    rows = []
    for case in load_cases():
        try:
            r = httpx.post(args.url.rstrip('/') + '/arena/run', json=case['request'], timeout=70)
            r.raise_for_status()
            row = score(case, r.json())
        except Exception as exc:
            row = {'name': case['name'], 'category': case['category'], 'passed': False,
                   'error': f'{type(exc).__name__}: {exc}'[:200]}
        rows.append(row)
        print(f"{'PASS' if row['passed'] else 'FAIL'}  {row['name']:<48} "
              f"{row.get('status', '-')}/{row.get('stop_reason', '-')}")
        time.sleep(args.pause)

    passed = sum(r['passed'] for r in rows)
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f'public_results_{args.label}.json'
    out.write_text(json.dumps({'url': args.url, 'passed': passed, 'total': len(rows),
                               'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                               'cases': rows}, indent=2), encoding='utf-8')
    print(f'{passed}/{len(rows)} public checks passed -> {out.relative_to(HERE.parent)}')
    sys.exit(0 if passed == len(rows) else 1)


if __name__ == '__main__':
    main()
