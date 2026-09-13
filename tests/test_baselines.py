import pytest

from p2.baselines import ConstantSpreadMM, SymmetricMM, compare_strategies
from p2.simulator import AVSSimulator


def test_symmetric_mm_quotes_symmetric() -> None:
    strategy = SymmetricMM(half_spread=0.5)
    bid, ask = strategy.quotes(100.0, 0, 0.0)
    assert bid == pytest.approx(99.5)
    assert ask == pytest.approx(100.5)


def test_constant_spread_no_time_decay() -> None:
    strategy = ConstantSpreadMM(sigma=1.0, gamma=0.1, kappa=1.5, T=1.0)
    bid_1, ask_1 = strategy.quotes(100.0, 0, 0.0)
    bid_2, ask_2 = strategy.quotes(100.0, 5, 0.8)
    assert ask_1 - bid_1 == pytest.approx(ask_2 - bid_2)


def test_avs_beats_symmetric() -> None:
    avs = AVSSimulator(sigma=2.0, gamma=0.5, kappa=1.5, A=120.0, T=0.5, dt=0.005, Q_max=8, seed=11)
    symmetric = SymmetricMM(half_spread=0.2)

    comparison = compare_strategies(
        {
            "avs": avs,
            "symmetric": (
                symmetric,
                {"sigma": 2.0, "kappa": 1.5, "A": 120.0, "T": 0.5, "dt": 0.005, "Q_max": 8, "seed": 11},
            ),
        },
        n_paths=1000,
    ).set_index("strategy")

    assert comparison.loc["avs", "sharpe"] > comparison.loc["symmetric", "sharpe"]
