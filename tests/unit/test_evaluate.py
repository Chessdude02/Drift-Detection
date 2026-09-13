from __future__ import annotations

import pytest

from pdm.training.evaluate import mae, passes_validation_gate, rmse

pytestmark = pytest.mark.unit


def test_rmse_zero_error():
    assert rmse([0, 0, 0], [0, 0, 0]) == 0.0


def test_rmse_known_value():
    # errors = [0, 10] -> mean squared error = 50 -> rmse = sqrt(50)
    assert rmse([0, 10], [0, 0]) == 50**0.5


def test_mae_known_value():
    assert mae([0, 10, 20], [0, 0, 0]) == 10.0


def test_passes_validation_gate_boundary():
    assert passes_validation_gate(20.0, max_rmse=20.0) is True
    assert passes_validation_gate(20.01, max_rmse=20.0) is False


def test_passes_validation_gate():
    assert passes_validation_gate(10.0, max_rmse=20.0) is True
    assert passes_validation_gate(30.0, max_rmse=20.0) is False
