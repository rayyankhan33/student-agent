const $ = (id) => document.getElementById(id);
let session = crypto.randomUUID();
let scenarios = [];
let busy = false;

function el(tag, cls, text) {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text !== undefined) n.textContent = text;
  return n;
}

function addMessage(role, text, status) {
  $('messages').querySelector('.hint')?.remove();
  const m = el('article', role);
  m.append(el('strong', '', role === 'user' ? 'You' : role === 'agent' ? 'Agent' : 'System'));
  if (status) m.append(el('span', 'tag ' + status, status));
  m.append(el('p', '', text));
  $('messages').append(m);
  m.scrollIntoView({ block: 'nearest' });
}

function setStatus(text, cls) {
  $('status').textContent = text;
  $('status').className = 'badge ' + (cls || '');
}

function renderTimeline(data) {
  const list = $('timeline');
  list.replaceChildren();
  for (const e of data.events || []) {
    let text = null;
    if (e.event === 'run_start') text = `start: model ${e.model}, max ${e.max_steps} steps, fault ${e.fault}, ${e.history_messages} history msgs${e.pending_goal ? ', pending goal' : ''}`;
    else if (e.event === 'llm_reply') text = `model reply (${e.model}, ${Math.round(e.latency_ms)} ms, ${e.input_tokens ?? '?'}/${e.output_tokens ?? '?'} tokens)`;
    else if (e.event === 'decision') text = `decision: ${e.status}${e.action ? ' → ' + e.action + ' ' + JSON.stringify(e.arguments) : ''}`;
    else if (e.event === 'validation') text = e.passed ? 'contract validation passed' : `contract violation (repair ${e.repairs_used}): ${e.detail}`;
    else if (e.event === 'observation') text = `observation [${e.tool}, ${e.attempts} attempt(s)] ${e.ok ? 'ok' : 'refused'}: ${e.summary}`;
    else if (e.event === 'policy_gate') text = `policy gate: ${e.tool} → ${e.verdict}`;
    else if (e.event === 'fault_injected') text = `fault injected: ${e.fault}`;
    else if (e.event === 'llm_error') text = `model call failed (${e.model}): ${e.error}`;
    else if (e.event === 'model_fallback') text = `failover: ${e.from_model} → ${e.to_model}`;
    else if (e.event === 'stop') text = `stop: ${e.status} (${e.stop_reason})`;
    if (text) list.append(el('li', e.event + (e.passed === false ? ' bad' : ''), `step ${e.step} · ${text}`));
  }
  const failed = (data.tool_calls || []).filter((t) => t.outcome !== 'success');
  for (const t of failed) list.append(el('li', 'warn', `tool trace: ${t.tool} attempt ${t.attempt} → ${t.outcome} (${Math.round(t.latency_ms)} ms)`));
  if (!list.children.length) list.append(el('li', 'muted', 'No events.'));

  const obs = $('observations');
  obs.replaceChildren();
  const items = (data.events || []).filter((e) => e.event === 'observation');
  if (!items.length) { obs.textContent = 'No tool was executed in this run.'; return; }
  for (const o of items) {
    const d = el('details', 'obs');
    d.append(el('summary', o.ok ? '' : 'warn', `step ${o.step} · ${o.tool}: ${o.summary}`));
    let pretty = o.data || '{}';
    try { pretty = JSON.stringify(JSON.parse(o.data), null, 2); } catch { /* truncated preview */ }
    d.append(el('pre', '', pretty));
    obs.append(d);
  }
}

async function refreshBoard() {
  try {
    const r = await fetch(`/chat/${session}/board`);
    const data = await r.json();
    const box = $('board');
    box.replaceChildren();
    if (!data.records.length) { box.textContent = 'Starts from the sample board after your first chat message.'; return; }
    const table = el('table');
    const head = el('tr');
    for (const h of ['id', 'company', 'role', 'status']) head.append(el('th', '', h));
    table.append(head);
    for (const rec of data.records) {
      const row = el('tr');
      for (const k of ['id', 'company', 'role', 'status']) row.append(el('td', '', rec[k]));
      table.append(row);
    }
    box.append(table);
    for (const d of data.outbox) box.append(el('p', 'muted', `${d.draft_id} for ${d.application_id} (${d.state}): “${d.subject}”`));
  } catch { /* board view is optional */ }
}

async function send(task, extra = {}) {
  const note = extra.external ?? $('external').value.trim();
  const external = extra.external_context ?? (note ? [{ source: $('source').value || 'pasted-note', content: note, trust: 'untrusted' }] : []);
  const arena_config = { max_steps: Number($('maxSteps').value) || 6, fault: $('fault').value };
  const stateless = $('stateless').checked;
  addMessage('user', task + (external.length ? `\n[+ ${external.length} untrusted context item(s)]` : '') +
    (arena_config.fault !== 'none' ? `\n[fault: ${arena_config.fault}]` : '') +
    (arena_config.max_steps !== 6 ? `\n[max_steps: ${arena_config.max_steps}]` : '') +
    (stateless ? '\n[stateless /arena/run]' : ''));
  busy = true;
  $('send').disabled = $('reset').disabled = $('runScenario').disabled = true;
  setStatus('running…', 'running');
  try {
    const url = stateless ? '/arena/run' : '/chat';
    const body = stateless ? { task, external_context: external, arena_config }
      : { session_id: session, task, model: $('model').value, external_context: external, arena_config };
    const r = await fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
    const data = await r.json();
    if (!r.ok) throw new Error(typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail));
    addMessage('agent', data.final_response, data.status);
    setStatus(`${data.status} · ${data.stop_reason}`, data.status);
    const m = data.metrics;
    $('meta').textContent = `${data.model ? 'model ' + data.model + ' · ' : 'stateless arena run · '}${data.steps} step(s) · ${m.model_calls} model call(s) · ` +
      `tokens ${m.input_tokens ?? '–'}/${m.output_tokens ?? '–'} · cost ≈ $${m.estimated_cost_usd ?? '–'} · ${Math.round(m.latency_ms)} ms` +
      (data.history_messages !== undefined ? ` · ${data.history_messages} messages in memory` : '') +
      (data.pending_goal ? ` · pending goal: “${data.pending_goal}”` : '');
    renderTimeline(data);
    $('raw').textContent = JSON.stringify(data, null, 2);
    if (!stateless) refreshBoard();
    return data;
  } catch (e) {
    setStatus('error: ' + e.message, 'failed');
    return null;
  } finally {
    busy = false;
    $('send').disabled = $('reset').disabled = false;
    $('runScenario').disabled = !$('scenario').value;
  }
}

$('chat').onsubmit = async (ev) => {
  ev.preventDefault();
  const task = $('task').value.trim();
  if (!task || busy) return;
  const data = await send(task);
  if (data) { $('task').value = ''; $('external').value = ''; }
};

function loadScenario() {
  const s = scenarios[Number($('scenario').value)];
  $('runScenario').disabled = !s;
  if (!s) { $('expected').textContent = ''; return; }
  $('task').value = s.turns[0];
  const ext = s.external_context[0];
  $('external').value = ext ? ext.content : '';
  $('source').value = ext ? ext.source : 'pasted-note';
  $('extDetails').open = !!ext;
  $('fault').value = s.arena_config.fault || 'none';
  $('maxSteps').value = s.arena_config.max_steps || 6;
  $('arenaDetails').open = (s.arena_config.fault && s.arena_config.fault !== 'none') || (s.arena_config.max_steps || 6) < 6;
  $('expected').textContent = `Category: ${s.category} · expected status: ${s.expected_status.join(' or ')}` +
    (s.turns.length > 1 ? ` · ${s.turns.length} turns: ${s.turns.map((t) => '“' + t + '”').join(' then ')}` : '');
}

$('scenario').onchange = loadScenario;
$('runScenario').onclick = async () => {
  const s = scenarios[Number($('scenario').value)];
  if (!s || busy) return;
  if (s.turns.length > 1) $('stateless').checked = false;   // memory is needed
  for (const turn of s.turns) {
    const data = await send(turn, { external_context: turn === s.turns[0] ? s.external_context : [] });
    if (!data) break;
  }
  $('task').value = '';
};

$('reset').onclick = async () => {
  await fetch('/chat/' + session, { method: 'DELETE' }).catch(() => {});
  session = crypto.randomUUID();
  $('messages').replaceChildren();
  $('timeline').replaceChildren(el('li', 'muted', 'No run yet.'));
  $('observations').textContent = 'No observations yet.';
  $('board').textContent = 'Loads after the first message.';
  $('raw').textContent = '—';
  $('meta').textContent = 'new session · 0 messages in memory';
  $('fault').value = 'none';
  $('maxSteps').value = 6;
  $('stateless').checked = false;
  setStatus('idle', 'idle');
};

async function init() {
  try {
    const r = await fetch('/models');
    const data = await r.json();
    for (const m of data.models) {
      const o = el('option', '', m + (m === data.default ? ' (default)' : ''));
      o.value = m;
      if (m === data.default) o.selected = true;
      $('model').append(o);
    }
  } catch { setStatus('models unavailable', 'failed'); }
  try {
    const r = await fetch('/demo/scenarios');
    scenarios = (await r.json()).scenarios;
    scenarios.forEach((s, i) => {
      const o = el('option', '', `[${s.category}] ${s.name}`);
      o.value = String(i);
      $('scenario').append(o);
    });
  } catch { /* scenarios are optional */ }
}

init();
