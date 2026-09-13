from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

import p2.research as research
from p2.bybit_replay import StrategySpec
from p2.config import load_config


def test_required_bybit_dates_cover_locked_windows() -> None:
    schedule = research.required_bybit_dates()

    assert tuple(schedule) == research.CRYPTO_SYMBOLS
    assert schedule["BTCUSDT"][0] == "2025-05-01"
    assert schedule["BTCUSDT"][-1] == "2026-04-27"
    assert len(schedule["BTCUSDT"]) == 362
    assert schedule["ETHUSDT"] == schedule["SOLUSDT"]
    assert len(schedule["ETHUSDT"]) == 86
    assert schedule["ETHUSDT"][0:2] == ("2025-07-01", "2025-07-02")
    assert schedule["ETHUSDT"][-2:] == ("2026-04-21", "2026-04-22")


def test_available_bybit_dates_requires_both_raw_files(tmp_path: Path) -> None:
    book_root = (
        tmp_path
        / "exchange=bybit"
        / "instrument_type=orderbook_l2"
        / "BTCUSDT"
    )
    trade_root = (
        tmp_path / "exchange=bybit" / "instrument_type=trades" / "BTCUSDT"
    )
    for value in ("2025-05-01", "2025-05-02"):
        path = book_root / f"date={value}" / "orderbook.parquet"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"book")
    trade = trade_root / "date=2025-05-02" / "trades.parquet"
    trade.parent.mkdir(parents=True)
    trade.write_bytes(b"trades")

    assert research.available_bybit_dates(tmp_path, "BTCUSDT") == (
        date.fromisoformat("2025-05-02"),
    )


def test_research_cache_defaults_to_external_volume(monkeypatch) -> None:
    monkeypatch.delenv("P2_CACHE_DIR", raising=False)
    config = load_config()
    config.paths.lake_dir = Path("/Volumes/Example Disk/data/market_data")

    assert research.research_cache_dir(config) == Path(
        "/Volumes/Example Disk/cache/p2"
    )


def test_cache_reconstruction_skips_existing_days(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cache_dir = tmp_path / "cache"
    existing = cache_dir / "bybit" / "BTCUSDT" / "2025-05-01.parquet"
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"cached")
    schedule = {
        "BTCUSDT": ("2025-05-01",),
        "ETHUSDT": ("2025-07-01",),
        "SOLUSDT": ("2025-07-01",),
    }
    calls: list[tuple[str, str]] = []

    def fake_find(root: Path, symbol: str, date_value: str) -> tuple[Path, Path]:
        del root
        calls.append((symbol, date_value))
        return Path("book.parquet"), Path("trades.parquet")

    def fake_reconstruct(
        orderbook: Path,
        trades: Path,
        destination: Path,
        symbol: str,
        date_value: str,
    ) -> None:
        del orderbook, trades
        path = destination / "bybit" / symbol / f"{date_value}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"reconstructed")

    monkeypatch.setattr(research, "find_bybit_files", fake_find)
    monkeypatch.setattr(research, "reconstruct_bybit_day", fake_reconstruct)

    summary = research.ensure_bybit_cache(
        tmp_path,
        cache_dir,
        schedule,
        workers=1,
    )

    assert summary == {
        "required_days": 3,
        "cached_days": 1,
        "reconstructed_days": 2,
        "workers": 1,
    }
    assert calls == [("ETHUSDT", "2025-07-01"), ("SOLUSDT", "2025-07-01")]


def test_research_runner_publishes_every_study(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = load_config()
    config.paths.lake_dir = tmp_path / "lake"
    config.paths.cache_dir = tmp_path / "cache"
    config.paths.lobster_dir = tmp_path / "lobster"
    specs = (StrategySpec("symmetric"), StrategySpec("glft", gamma=0.0001))
    queue_table = pd.DataFrame([{"method": "exact_fifo"}])
    daily = pd.DataFrame([{"strategy": "symmetric"}])
    decomposition = pd.DataFrame([{"strategy": "symmetric"}])
    markouts = pd.DataFrame([{"strategy": "symmetric"}])
    crosscheck = pd.DataFrame([{"date": "a"}, {"date": "b"}])
    appendix = pd.DataFrame([{"symbol": "AAPL"}])
    published: dict[str, object] = {}

    monkeypatch.setattr(
        research,
        "ensure_bybit_cache",
        lambda source, cache: {
            "required_days": 1,
            "cached_days": 1,
            "reconstructed_days": 0,
            "workers": 1,
        },
    )
    monkeypatch.setattr(research, "find_mbo_file", lambda root, symbol: Path("es"))
    monkeypatch.setattr(
        research,
        "validate_es_queues",
        lambda source: SimpleNamespace(selected_rule="proportional", table=queue_table),
    )
    monkeypatch.setattr(
        research,
        "run_crypto_study",
        lambda cache, cancellation_rule, workers: SimpleNamespace(
            summary={"main_study": []},
            daily_pnl=daily,
            decomposition=decomposition,
            markouts=markouts,
            selection=SimpleNamespace(strategies=specs),
        ),
    )
    monkeypatch.setattr(
        research,
        "run_hft_crosscheck",
        lambda cache, native, selected: crosscheck,
    )
    monkeypatch.setattr(
        research,
        "run_lobster_appendix",
        lambda data, selected, cancellation_rule: SimpleNamespace(
            table=appendix,
            included_symbols=("AAPL",),
            missing_symbols=("AMZN", "GOOG", "INTC", "MSFT"),
        ),
    )

    def fake_publish(destination: Path, **tables: object) -> None:
        published["destination"] = destination
        published.update(tables)

    monkeypatch.setattr(research, "publish_research_outputs", fake_publish)

    result = research.run_research(config, output_dir=tmp_path / "published")

    assert result.output_dir == tmp_path / "published"
    assert result.summary["queue_validation"]["selected_rule"] == "proportional"
    assert result.summary["external_crosscheck"]["row_count"] == 2
    assert result.summary["lobster_appendix"]["row_count"] == 1
    assert published["daily_pnl"] is daily
    assert published["queue_validation"] is queue_table
    assert published["hftbacktest_crosscheck"] is crosscheck
    assert published["lobster_appendix"] is appendix
