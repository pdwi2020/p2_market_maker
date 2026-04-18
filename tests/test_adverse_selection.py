import numpy as np

from p2.adverse_selection import ArrivalModel, calibrate_epsilon


def test_arrival_intensity_decays_with_distance() -> None:
    near = ArrivalModel.intensity(0.1, A=100.0, kappa=1.5)
    far = ArrivalModel.intensity(1.0, A=100.0, kappa=1.5)
    assert near > far


def test_adverse_jump_sign_convention() -> None:
    assert ArrivalModel.adverse_jump("ask", 0.05) == 0.05
    assert ArrivalModel.adverse_jump("bid", 0.05) == -0.05
    assert ArrivalModel.adverse_jump(-1, 0.05) == 0.05
    assert ArrivalModel.adverse_jump(1, 0.05) == -0.05


def test_calibrate_epsilon_from_fixture(tmp_path) -> None:
    path = tmp_path / "prices.csv"
    path.write_text("price\n100.0\n100.1\n100.3\n")
    epsilon = calibrate_epsilon(path)
    assert np.isclose(epsilon, 0.075)
