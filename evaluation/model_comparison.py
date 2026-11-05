"""Model-selection experiment: run the same public cases against several models
in-process and record success, contract validity, action selection, latency,
tokens and approximate cost.

  python -m evaluation.model_comparison --models gemini-3.1-flash-lite,gemini-3.5-flash-lite

Writes evaluation/results/model_comparison.json and model_comparison.md.
"""
import argparse
import asyncio
import json
import time
from statistics import mean

from app.arena import execute
from app.config import settings
from app.models import ArenaRequest
from evaluation.run_public_tests import RESULTS, load_cases, score


async def run_model(model: str, cases: list[dict], pause: float) -> list[dict]:
    rows = []
    for case in cases:
        req = ArenaRequest.model_validate(case['request'])
        provider_retries = 0
        while True:
            result = (await execute(req, model=model)).model_dump()
            # A provider outage / rate limit says nothing about the model's quality:
            # wait and re-run the case (recorded) instead of scoring it as a failure.
            if result['stop_reason'] != 'llm_unavailable' or provider_retries >= 3:
                break
            provider_retries += 1
            print(f'  [{model}] provider unavailable, retry {provider_retries} after 30 s')
            await asyncio.sleep(30)
        row = score(case, result)
        injected = 1 if case['request']['arena_config'].get('fault') == 'invalid_agent_decision' else 0
        row['model_calls'] = result['metrics']['model_calls']
        row['provider_retries'] = provider_retries
        row['natural_violations'] = max(0, row['contract_violations'] - injected)
        rows.append(row)
        print(f"  [{model}] {'PASS' if row['passed'] else 'FAIL'} {case['name']} -> "
              f"{row['status']} ({row['metrics']['latency_ms']:.0f} ms)")
        await asyncio.sleep(pause)
    return rows


def summarise(model: str, rows: list[dict]) -> dict:
    calls = sum(r['model_calls'] for r in rows) or 1
    action_rows = [r for r in rows if r['action_correct'] is not None]
    m = [r['metrics'] for r in rows]
    cost = [x['estimated_cost_usd'] for x in m if x.get('estimated_cost_usd') is not None]
    return {
        'model': model,
        'cases': len(rows),
        'task_success': sum(r['passed'] for r in rows),
        'structured_output_validity': round(1 - sum(r['natural_violations'] for r in rows) / calls, 3),
        'action_selection': f"{sum(r['action_correct'] for r in action_rows)}/{len(action_rows)}",
        'avg_latency_ms': round(mean(x['latency_ms'] for x in m)),
        'avg_model_calls': round(calls / len(rows), 2),
        'avg_input_tokens': round(mean((x.get('input_tokens') or 0) for x in m)),
        'avg_output_tokens': round(mean((x.get('output_tokens') or 0) for x in m)),
        'avg_cost_usd': round(mean(cost), 6) if cost else None,
        'price_per_1m_in_out': settings.price(model),
        'provider_retries': sum(r['provider_retries'] for r in rows),
    }


def to_markdown(summaries: list[dict]) -> str:
    head = ('| Model | Task success | Structured-output validity | Correct action | '
            'Avg latency | Avg calls/run | Avg tokens in/out | Approx. cost/run |\n'
            '|---|---|---|---|---|---|---|---|\n')
    rows = ''.join(
        f"| {s['model']} | {s['task_success']}/{s['cases']} | {s['structured_output_validity']:.1%} | "
        f"{s['action_selection']} | {s['avg_latency_ms']} ms | {s['avg_model_calls']} | "
        f"{s['avg_input_tokens']}/{s['avg_output_tokens']} | "
        f"{'$%.5f' % s['avg_cost_usd'] if s['avg_cost_usd'] is not None else 'n/a'} |\n"
        for s in summaries)
    return head + rows


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', default='gemini-3.1-flash-lite,gemini-3.5-flash-lite')
    parser.add_argument('--pause', type=float, default=12.0, help='seconds between cases (rate limits)')
    args = parser.parse_args()
    settings.model_fallback = False      # measure each model on its own
    cases = load_cases()
    detail, summaries = {}, []
    for model in [m.strip() for m in args.models.split(',') if m.strip()]:
        print(f'== {model}')
        rows = await run_model(model, cases, args.pause)
        detail[model] = rows
        summaries.append(summarise(model, rows))
    RESULTS.mkdir(exist_ok=True)
    stamp = time.strftime('%Y-%m-%d %H:%M:%S')
    (RESULTS / 'model_comparison.json').write_text(json.dumps(
        {'generated_at': stamp, 'summary': summaries, 'cases': detail}, indent=2), encoding='utf-8')
    md = f'# Model comparison ({stamp})\n\n{to_markdown(summaries)}'
    (RESULTS / 'model_comparison.md').write_text(md, encoding='utf-8')
    print(md)


if __name__ == '__main__':
    asyncio.run(main())
