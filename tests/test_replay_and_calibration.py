from pathlib import Path

from p2.backtest import pnl_attribution, run_lobster_backtest, write_backtest_outputs
from p2.calibration import calibrate_from_replay_files, write_calibration
from p2.config import load_config
from p2.lobster_replay import LOBSTERReplayer


FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"


def test_lobster_replayer_loads_fixture() -> None:
    replayer = LOBSTERReplayer().load(
        FIXTURE_DIR / "lobster_orderbook.csv",
        FIXTURE_DIR / "lobster_message.csv",
    )
    mids = replayer.mid_price_series()
    assert len(mids) == 4
    assert mids.iloc[0] == 100.0


def test_calibration_from_fixture_is_non_negative() -> None:
    calibration = calibrate_from_replay_files(
        FIXTURE_DIR / "lobster_orderbook.csv",
        FIXTURE_DIR / "lobster_message.csv",
    )
    assert calibration["sigma"] >= 0.0
    assert calibration["A"] >= 0.0
    assert calibration["kappa"] > 0.0
    assert calibration["epsilon"] >= 0.0


def test_backtest_runs_on_fixture(tmp_path: Path) -> None:
    config = load_config()
    config.paths.results_dir = tmp_path
    config.replay.orderbook_file = str(FIXTURE_DIR / "lobster_orderbook.csv")
    config.replay.message_file = str(FIXTURE_DIR / "lobster_message.csv")

    result = run_lobster_backtest(config)
    summary = pnl_attribution(result)
    assert "terminal_pnl" in summary
    assert summary["n_fills"] >= 0

    run_dir = write_backtest_outputs(config, result)
    assert (run_dir / "replay_summary.json").exists()
    assert (run_dir / "replay_inventory.csv").exists()

    calibration_path = write_calibration(
        config,
        calibrate_from_replay_files(
            FIXTURE_DIR / "lobster_orderbook.csv",
            FIXTURE_DIR / "lobster_message.csv",
        ),
    )
    assert calibration_path.exists()
