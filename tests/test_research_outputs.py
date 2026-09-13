import json
from pathlib import Path

import pandas as pd

from p2.research_outputs import FIGURE_FILES, TABLE_FILES, publish_research_outputs


def test_publication_contract_writes_tables_and_figures(tmp_path: Path) -> None:
    daily = pd.DataFrame(
        [
            {
                "symbol": "BTCUSDT",
                "date": "2025-07-01",
                "strategy": "symmetric",
                "maker_fee_rate": 0.0002,
                "latency_ms": 50,
                "net_pnl": 1.25,
                "end_inventory": 0.01,
            },
            {
                "symbol": "BTCUSDT",
                "date": "2025-07-02",
                "strategy": "symmetric",
                "maker_fee_rate": 0.0002,
                "latency_ms": 50,
                "net_pnl": -0.25,
                "end_inventory": 0.0,
            },
        ]
    )
    decomposition = pd.DataFrame(
        [
            {
                "symbol": "BTCUSDT",
                "strategy": "symmetric",
                "maker_fee_rate": 0.0002,
                "latency_ms": 50,
                "realized_spread": 2.0,
                "inventory_revaluation": -0.5,
                "fee_cost": 0.5,
            }
        ]
    )
    markouts = pd.DataFrame(
        [
            {
                "symbol": "BTCUSDT",
                "strategy": "symmetric",
                "maker_fee_rate": 0.0002,
                "latency_ms": 50,
                "horizon_seconds": 1,
                "mean_markout": -0.01,
            }
        ]
    )
    queue = pd.DataFrame(
        [
            {
                "method": "proportional",
                "fill_count_relative_bias": 0.1,
                "fill_time_ks": 0.2,
                "gross_pnl_error": -1.0,
            }
        ]
    )

    publish_research_outputs(
        tmp_path,
        summary={"runtime_seconds": 1.0},
        daily_pnl=daily,
        decomposition=decomposition,
        markouts=markouts,
        queue_validation=queue,
        hftbacktest_crosscheck=pd.DataFrame(columns=["date", "strategy"]),
        lobster_appendix=pd.DataFrame(columns=["symbol", "strategy"]),
    )

    assert set(TABLE_FILES.values()).issubset(path.name for path in tmp_path.iterdir())
    assert set(FIGURE_FILES).issubset(path.name for path in tmp_path.iterdir())
    assert json.loads((tmp_path / "summary.json").read_text())["runtime_seconds"] == 1.0
    assert all((tmp_path / filename).stat().st_size > 0 for filename in FIGURE_FILES)
