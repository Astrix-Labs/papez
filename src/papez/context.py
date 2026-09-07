"""Request-scoped user and org identity via contextvars."""
from __future__ import annotations

import contextvars

current_user_id: contextvars.ContextVar[str] = contextvars.ContextVar("current_user_id")
current_org_id: contextvars.ContextVar[str | None] = contextvars.ContextVar("current_org_id", default=None)
# No default on purpose: a shared mutable list default is a cross-request
# leak hazard. Readers call current_org_ids.get([]) for a fresh empty list.
current_org_ids: contextvars.ContextVar[list[str]] = contextvars.ContextVar("current_org_ids")
current_user_role: contextvars.ContextVar[str | None] = contextvars.ContextVar("current_user_role", default=None)
