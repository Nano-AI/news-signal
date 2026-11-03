"""Wire-protocol models. `docs/protocol.md` is the prose contract."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

Source = Literal["benzinga", "edgar"]
Timestamp = Annotated[str, Field(pattern=r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")]


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
    reason: Literal["unknown_story", "unknown_ticker", "not_yet_available"]


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
    tickers: list[str]
    headline: str
    summary: str
    returns: dict[str, float | None]
