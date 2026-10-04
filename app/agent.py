"""Bounded agent loop for Job-Application Tracker.

This module implements the core agentic loop:
  1. Assemble prompt (system policy + history + state + user goal + external context)
  2. Call the LLM
  3. Parse and validate the AgentDecision (schema + semantic)
  4. Execute tool or stop
  5. Feed result back as observation
  6. Repeat until done or budget exhausted

All model calls (including repair attempts) count against max_steps.
"""
import json
import logging
from time import perf_counter

from app.config import settings
from app.models import (
    AgentDecision, AgentState, ArenaResponse, Metrics, ToolTrace
)
from app.prompts import build_messages
from app.tools import (
    execute_tool, get_board_snapshot, TOOLS, BLOCKED_ACTIONS
)

log = logging.getLogger('agent')


def _get_llm(model: str):
    """Get a LangChain LLM instance for the given model name."""
    provider = settings.model_provider.lower()

    if provider == 'gemini' or provider == 'google':
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=settings.gemini_api_key,
            temperature=0.1,
            max_output_tokens=settings.max_output_tokens,
        )
    else:
        raise ValueError(
            f"Unsupported model provider: '{provider}'. "
            f"Set MODEL_PROVIDER to 'gemini'."
        )


def _parse_decision(raw_text: str) -> AgentDecision:
    """Parse and validate the LLM's raw text output into an AgentDecision.

    Handles cases where the model wraps JSON in markdown code fences.
    """
    text = raw_text.strip()

    # Strip markdown code fences if present
    if text.startswith('```'):
        lines = text.split('\n')
        # Remove first line (```json or ```) and last line (```)
        if lines[-1].strip() == '```':
            lines = lines[1:-1]
        else:
            lines = lines[1:]
        text = '\n'.join(lines).strip()

    # Try to find JSON object in the text
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1 and end > start:
        text = text[start:end + 1]

    data = json.loads(text)
    return AgentDecision.model_validate(data)


def _inject_invalid_decision_fault(raw_text: str) -> str:
    """Corrupt the model output to simulate an invalid_agent_decision fault."""
    return '{"status": "INVALID_ENUM", "action": 999, "broken": true}'


async def run_agent(request, history, model) -> ArenaResponse:
    """Main bounded agent loop.

    Args:
        request: ArenaRequest with task, external_context, arena_config
        history: list of LangChain messages from prior conversation turns
        model: model name string

    Returns:
        ArenaResponse with status, final_response, steps, traces, etc.
    """
    state = AgentState(max_steps=request.arena_config.max_steps)
    fault_type = request.arena_config.fault.type
    fault_triggered = False  # Track if fault has been injected

    # Check if model is configured
    if model == 'unconfigured' or settings.model_provider == 'unconfigured':
        return ArenaResponse(
            request_id=request.request_id,
            status='failed',
            final_response='No model is configured. Set MODEL_PROVIDER and MODEL_NAME.',
            stop_reason='no_model_configured',
            events=[{'step': 0, 'event': 'agent_stop',
                     'reason': 'no_model_configured'}]
        )

    try:
        llm = _get_llm(model)
    except Exception as e:
        return ArenaResponse(
            request_id=request.request_id,
            status='failed',
            final_response=f'Failed to initialize model: {str(e)[:200]}',
            stop_reason='model_init_error',
            events=[{'step': 0, 'event': 'agent_stop',
                     'reason': f'model_init_error: {str(e)[:100]}'}]
        )

    # -------------------------------------------------------------------
    # Main agent loop
    # -------------------------------------------------------------------
    while state.step_count < state.max_steps:
        state.step_count += 1
        step = state.step_count

        state.events.append({
            'step': step, 'event': 'loop_start',
            'remaining': state.max_steps - step
        })

        # 1. Assemble prompt
        board_snapshot = get_board_snapshot()
        messages = build_messages(
            task=request.task,
            history=history,
            external_context=request.external_context,
            observations=state.observations,
            board_snapshot=board_snapshot,
            step=step,
            max_steps=state.max_steps,
        )

        # 2. Call the LLM
        try:
            llm_response = await llm.ainvoke(messages)
            raw_text = llm_response.content
            state.model_calls += 1

            # Track token usage if available
            usage = getattr(llm_response, 'usage_metadata', None)
            if usage:
                state.total_input_tokens += getattr(
                    usage, 'input_tokens', 0) or 0
                state.total_output_tokens += getattr(
                    usage, 'output_tokens', 0) or 0

            state.events.append({
                'step': step, 'event': 'llm_response',
                'raw_length': len(raw_text),
            })

        except Exception as e:
            log.error('LLM call failed at step %d: %s', step, e)
            state.errors.append({
                'step': step, 'error': 'llm_call_failed',
                'detail': str(e)[:200]
            })
            state.events.append({
                'step': step, 'event': 'llm_error',
                'error': str(e)[:100]
            })
            # Try to continue if we have budget, otherwise fail
            if state.step_count >= state.max_steps:
                return _build_response(
                    request.request_id, 'failed',
                    'The model call failed and no budget remains to retry.',
                    'llm_error', state
                )
            continue

        # 3. Parse and validate the decision
        # Optionally inject invalid_agent_decision fault
        if (fault_type == 'invalid_agent_decision'
                and not fault_triggered):
            raw_text = _inject_invalid_decision_fault(raw_text)
            fault_triggered = True
            state.events.append({
                'step': step, 'event': 'fault_injected',
                'fault': 'invalid_agent_decision'
            })

        decision = None
        parse_error = None
        try:
            decision = _parse_decision(raw_text)
        except Exception as e:
            parse_error = str(e)

        # If parsing failed, attempt one repair (counts as another step)
        if decision is None:
            state.errors.append({
                'step': step, 'error': 'contract_parse_error',
                'detail': parse_error[:200] if parse_error else 'unknown'
            })
            state.events.append({
                'step': step, 'event': 'contract_violation',
                'detail': parse_error[:100] if parse_error else 'unknown'
            })

            # Check if we have budget for a repair attempt
            if state.step_count >= state.max_steps:
                return _build_response(
                    request.request_id, 'contract_error',
                    'The model produced invalid output and no budget remains '
                    'to retry.',
                    'contract_repair_exhausted', state
                )

            # Repair attempt — re-call LLM with error feedback
            state.step_count += 1
            state.observations.append(
                f"[REPAIR NEEDED] Your previous response was not valid JSON "
                f"matching the required schema. Error: {parse_error[:200]}. "
                f"Please respond with ONLY a valid JSON object matching the "
                f"AgentDecision schema."
            )

            try:
                repair_messages = build_messages(
                    task=request.task,
                    history=history,
                    external_context=request.external_context,
                    observations=state.observations,
                    board_snapshot=board_snapshot,
                    step=state.step_count,
                    max_steps=state.max_steps,
                )
                repair_response = await llm.ainvoke(repair_messages)
                raw_text = repair_response.content
                state.model_calls += 1

                usage = getattr(repair_response, 'usage_metadata', None)
                if usage:
                    state.total_input_tokens += getattr(
                        usage, 'input_tokens', 0) or 0
                    state.total_output_tokens += getattr(
                        usage, 'output_tokens', 0) or 0

                decision = _parse_decision(raw_text)
                state.events.append({
                    'step': state.step_count, 'event': 'contract_repaired'
                })

            except Exception as e2:
                state.errors.append({
                    'step': state.step_count, 'error': 'contract_repair_failed',
                    'detail': str(e2)[:200]
                })
                return _build_response(
                    request.request_id, 'contract_error',
                    'The model failed to produce valid output after a repair '
                    'attempt.',
                    'contract_repair_exhausted', state
                )

        # 4. Act on the decision
        state.events.append({
            'step': state.step_count,
            'event': 'decision',
            'status': decision.status,
            'action': decision.action,
        })

        # Terminal statuses — stop the loop
        if decision.status in ('completed', 'needs_clarification',
                               'blocked', 'failed', 'approval_required'):
            # Map agent decision status to arena response status
            arena_status = decision.status
            stop_reason = f'agent_{decision.status}'

            return _build_response(
                request.request_id,
                arena_status,
                decision.user_message or f'Agent stopped: {decision.status}',
                stop_reason,
                state
            )

        # Status is "continue" — execute the tool
        if decision.status == 'continue':
            if not decision.action:
                state.errors.append({
                    'step': state.step_count,
                    'error': 'no_action_with_continue',
                    'detail': 'Decision status is "continue" but no action '
                              'was specified.'
                })
                state.observations.append(
                    '[ERROR] You set status="continue" but did not specify '
                    'an action. Either provide an action or use a terminal '
                    'status.'
                )
                continue

            # Check autonomy boundary
            if decision.action in BLOCKED_ACTIONS:
                return _build_response(
                    request.request_id,
                    'approval_required',
                    f"The action '{decision.action}' is outside the agent's "
                    f"permitted authority. This would require human approval.",
                    'autonomy_boundary',
                    state
                )

            # Execute the tool through the fault-injection gateway
            try:
                tool_result, trace, newly_triggered = await execute_tool(
                    tool_name=decision.action,
                    arguments=decision.arguments,
                    step=min(state.step_count, 6),
                    fault_type=fault_type,
                    fault_triggered=fault_triggered,
                    max_retries=settings.max_tool_retries,
                )
                if newly_triggered:
                    fault_triggered = True

                state.tool_traces.append(trace)

                # Check for malformed tool output
                if isinstance(tool_result, dict) and tool_result.get(
                        '__malformed__'):
                    state.observations.append(
                        f"[TOOL ERROR] Tool '{decision.action}' returned "
                        f"malformed output. The data could not be parsed. "
                        f"You may retry or choose a different approach."
                    )
                    state.events.append({
                        'step': state.step_count,
                        'event': 'tool_malformed_output',
                        'tool': decision.action
                    })
                elif isinstance(tool_result, dict) and not tool_result.get(
                        'success', True):
                    error_msg = tool_result.get('error', 'Unknown tool error')
                    state.observations.append(
                        f"[TOOL RESULT] {decision.action}: Failed — {error_msg}"
                    )
                    state.events.append({
                        'step': state.step_count,
                        'event': 'tool_error',
                        'tool': decision.action,
                        'error': error_msg[:100]
                    })
                else:
                    # Successful tool result
                    result_summary = json.dumps(tool_result, default=str)
                    if len(result_summary) > 2000:
                        result_summary = result_summary[:2000] + '...'
                    state.observations.append(
                        f"[TOOL RESULT] {decision.action}: {result_summary}"
                    )
                    state.events.append({
                        'step': state.step_count,
                        'event': 'tool_success',
                        'tool': decision.action,
                    })

            except Exception as e:
                state.errors.append({
                    'step': state.step_count, 'error': 'tool_execution_error',
                    'tool': decision.action, 'detail': str(e)[:200]
                })
                state.observations.append(
                    f"[TOOL ERROR] {decision.action} raised an exception: "
                    f"{str(e)[:200]}"
                )
                state.events.append({
                    'step': state.step_count,
                    'event': 'tool_exception',
                    'tool': decision.action,
                    'error': str(e)[:100]
                })

    # -------------------------------------------------------------------
    # Budget exhausted — loop exited naturally
    # -------------------------------------------------------------------
    return _build_response(
        request.request_id,
        'budget_exceeded',
        state.observations[-1][:500] if state.observations
        else 'Step budget exhausted before the task completed.',
        'step_budget_reached',
        state
    )


def _build_response(
    request_id: str,
    status: str,
    final_response: str,
    stop_reason: str,
    state: AgentState,
) -> ArenaResponse:
    """Construct a valid ArenaResponse from the current agent state."""
    # Ensure final_response is not empty
    if not final_response or not final_response.strip():
        final_response = f'Agent stopped with status: {status}'

    # Truncate if needed
    if len(final_response) > 1900:
        final_response = final_response[:1900] + '...'

    return ArenaResponse(
        request_id=request_id,
        status=status,
        final_response=final_response,
        steps=min(state.step_count, 6),
        stop_reason=stop_reason,
        tool_calls=state.tool_traces,
        errors=state.errors,
        events=state.events,
        metrics=Metrics(
            model_calls=min(state.model_calls, 6),
            input_tokens=state.total_input_tokens or None,
            output_tokens=state.total_output_tokens or None,
            estimated_cost_usd=None,  # Free tier — no cost
        ),
    )
