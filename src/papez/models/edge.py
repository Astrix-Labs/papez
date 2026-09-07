from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

from papez.models.enums import EdgeType


class MemoryEdge(BaseModel):
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    source_id: uuid.UUID
    target_id: uuid.UUID
    type: EdgeType
    weight: float = 0.7
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    reason: str | None = None
    created_by: str | None = None
    source_context: str | None = None
    last_validated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] | None = None
