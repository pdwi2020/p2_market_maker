from pathlib import Path

import numpy as np
import pytest

from p2.baselines import AvSOptimalMM, SymmetricMM
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


def test_replay_passes_elapsed_session_time_to_strategy() -> None:
    replayer = LOBSTERReplayer().load(
        FIXTURE_DIR / "lobster_orderbook.csv",
        FIXTURE_DIR / "lobster_message.csv",
    )
    observed_times: list[float] = []

    def strategy(mid: float, inventory: int, t_session: float) -> tuple[float, float]:
        del inventory
        observed_times.append(t_session)
        return mid - 0.1, mid + 0.1

    replayer.run_strategy(strategy, session_duration_seconds=23_400.0)

    assert observed_times == pytest.approx([0.0, 0.1, 0.2, 0.3])
    assert np.all(23_400.0 - np.asarray(observed_times) > 0.0)


def test_inventory_aware_quotes_differ_from_symmetric_during_session() -> None:
    sigma = 0.04428732059348843
    gamma = 0.0001
    kappa = 25.37371953663545
    horizon = 23_400.0
    inventory = 5
    t_session = 1_800.0
    avs = AvSOptimalMM(sigma=sigma, gamma=gamma, kappa=kappa, T=horizon)
    symmetric = SymmetricMM(
        half_spread=0.5 * (avs.quotes(100.0, 0, t_session)[1] - avs.quotes(100.0, 0, t_session)[0])
    )

    avs_bid, avs_ask = avs.quotes(100.0, inventory, t_session)
    symmetric_bid, symmetric_ask = symmetric.quotes(100.0, inventory, t_session)

    assert avs_bid < symmetric_bid
    assert avs_ask < symmetric_ask


@pytest.mark.parametrize("use_queue_position", [False, True])
def test_execution_direction_uses_resting_order_side(tmp_path: Path, use_queue_position: bool) -> None:
    orderbook_file = tmp_path / "direction_orderbook.csv"
    message_file = tmp_path / "direction_message.csv"
    orderbook_file.write_text("\n".join(["1010000,1,990000,1"] * 6))
    message_file.write_text(
        "\n".join(
            [
                "34200.000,1,1,1,1000000,1",
                "34200.001,4,2,2,990000,1",
                "34200.002,1,3,1,1000000,1",
                "34200.003,4,4,2,1010000,-1",
                "34200.004,2,5,1,990000,1",
                "34200.005,2,6,1,1010000,-1",
            ]
        )
    )
    replayer = LOBSTERReplayer().load(orderbook_file, message_file)

    result = replayer.run_strategy(
        lambda mid, inventory, t: (99.0, 101.0),
        use_queue_position=use_queue_position,
    )

    assert len(result.fill_times) == 2
    assert result.inventory_path[2] == 1
    assert result.inventory_path[4] == 0
    assert result.inventory_path[-1] == 0
