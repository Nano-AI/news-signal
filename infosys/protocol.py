"""Wire-protocol models. `docs/protocol.md` is the prose contract."""

from datetime import datetime
from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator

Source = Literal["benzinga", "edgar"]


def _real_timestamp(value: str) -> str:
    datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")  # rejects impossible dates and times
    return value


Timestamp = Annotated[str, Field(pattern=r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$"), AfterValidator(_real_timestamp)]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class FeedInfo(_Model):
    name: str
    transport: Literal["ws", "poll"]
    path: str


class FeedList(_Model):
    feeds: list[FeedInfo]


class StoryMessage(_Model):
    type: Literal["story"]
    feed: str
    seq: int = Field(ge=1)
    story_id: str
    source: Source
    published_at: Timestamp
    tickers: list[str] = Field(min_length=1, max_length=5)
    headline: str
    summary: str


class EndMessage(_Model):
    type: Literal["end"]


class PollResponse(_Model):
    messages: list[StoryMessage]
    next: int = Field(ge=0)
    end: bool


class Signal(_Model):
    story_id: str
    ticker: str
    direction: Literal[-1, 0, 1]
    confidence: float = Field(ge=0, le=1, allow_inf_nan=False)


class SignalBatch(_Model):
    signals: list[Signal] = Field(min_length=1, max_length=1024)


class Rejection(_Model):
    index: int = Field(ge=0)
    reason: Literal["not_yet_available", "unknown_ticker"]


class SignalReply(_Model):
    accepted: int = Field(ge=0)
    rejected: list[Rejection]


class Status(_Model):
    state: Literal["waiting", "warmup", "running", "draining", "done"]
    released: int = Field(ge=0)
    total: int = Field(ge=0)


class TrainRecord(_Model):
    story_id: str
    source: Source
    published_at: Timestamp
    tickers: list[str] = Field(min_length=1, max_length=5)
    headline: str
    summary: str
    returns: dict[str, float | None]

    @model_validator(mode="after")
    def _returns_cover_tickers(self) -> "TrainRecord":
        if set(self.returns) != set(self.tickers):
            raise ValueError("returns keys must equal tickers")
        return self
