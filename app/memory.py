"""Per-session conversation memory (in-process, single worker).

Each session keeps:
  * messages — LangChain HumanMessage/AIMessage objects, bounded by count and characters
  * pending  — the unfinished goal while the agent is waiting for a clarification
  * board    — the session's own sandbox job board

Everything here is lost when the process restarts. That is acceptable for Assignment 1
and is documented in the README.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from app.config import settings
from app.tools import Board


@dataclass
class Session:
    messages: list[BaseMessage] = field(default_factory=list)
    pending: Optional[dict] = None
    board: Board = field(default_factory=Board.from_sample)


class Memory:
    def __init__(self, max_messages: int | None = None, max_chars: int | None = None,
                 max_sessions: int | None = None):
        self.max_messages = max_messages or settings.max_history_messages
        self.max_chars = max_chars or settings.max_history_chars
        self.max_sessions = max_sessions or settings.max_sessions
        self.sessions: OrderedDict[str, Session] = OrderedDict()

    def session(self, session_id: str) -> Session:
        if session_id not in self.sessions:
            self.sessions[session_id] = Session()
            while len(self.sessions) > self.max_sessions:
                self.sessions.popitem(last=False)      # evict least-recently used
        self.sessions.move_to_end(session_id)
        return self.sessions[session_id]

    def get(self, session_id: str) -> list[BaseMessage]:
        s = self.sessions.get(session_id)
        return list(s.messages) if s else []

    def add(self, session_id: str, user: str, assistant: str, status: str = 'completed'):
        s = self.session(session_id)
        msgs = s.messages + [HumanMessage(content=user), AIMessage(content=assistant)]
        # Drop whole oldest turns until both the count and character ceilings hold.
        while len(msgs) > self.max_messages or (
                len(msgs) > 2 and sum(len(str(m.content)) for m in msgs) > self.max_chars):
            msgs = msgs[2:]
        s.messages = msgs
        if status == 'needs_clarification':
            goal = s.pending['goal'] if s.pending else user
            if s.pending and user not in goal:
                goal = f'{goal} | user added: {user}'
            s.pending = {'goal': goal[:800], 'question': assistant[:400]}
        else:
            s.pending = None

    def clear(self, session_id: str):
        self.sessions.pop(session_id, None)
