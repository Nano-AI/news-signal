"""The ticker universe shared by data building, replay and scoring."""

TICKERS: tuple[str, ...] = (
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "TSLA", "AVGO", "JPM", "V", "MA",
    "UNH", "XOM", "JNJ", "WMT", "PG", "HD", "COST", "BAC", "NFLX", "AMD",
    "CRM", "ORCL", "KO", "PEP", "LLY",
)  # fmt: skip

MAX_TICKERS_PER_STORY = 5
