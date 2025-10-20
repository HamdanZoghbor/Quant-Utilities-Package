import pytest
import math
from quantmath.metrics  import (  # adjust import path as needed
    arithmetic_mean,
    weighted_mean,
    geometric_mean,
    population_variance,
    sharpe_ratio,
)

# ---- arithmetic_mean ----
def test_arithmetic_mean_basic():
    assert arithmetic_mean([1.0, 2.0, 3.0]) == 2.0

def test_arithmetic_mean_empty():
    with pytest.raises(ValueError):
        arithmetic_mean([])

def test_arithmetic_mean_nan():
    with pytest.raises(ValueError):
        arithmetic_mean([1.0, float("nan")])


# ---- weighted_mean ----
def test_weighted_mean_happy():
    result = weighted_mean([10, 20, 30], [1, 2, 3])
    assert math.isclose(result, 23.3333333, rel_tol=1e-6)

def test_weighted_mean_length_mismatch():
    with pytest.raises(ValueError):
        weighted_mean([10, 20], [1, 2, 3])

def test_weighted_mean_zero_weights():
    with pytest.raises(ValueError):
        weighted_mean([10, 20, 30], [0, 0, 0])


def test_geometric_mean_happy():
    assert geometric_mean([1, 2, 4]) == 2.0

def test_geometric_mean_contains_zero_or_negative():
    with pytest.raises(ValueError):
        geometric_mean([0, 2, 3])
    with pytest.raises(ValueError):
        geometric_mean([-1, 2, 3])

def test_geometric_mean_empty():
    with pytest.raises(ValueError):
        geometric_mean([])


# ---- population_variance ----
def test_population_variance_happy():
    result = population_variance([1, 2, 3])
    assert math.isclose(result, 2/3, rel_tol=1e-6)

def test_population_variance_empty_or_nan():
    with pytest.raises(ValueError):
        population_variance([])


# ---- sharpe_ratio ----
def test_sharpe_ratio_happy():
    result = sharpe_ratio([1, 2, 3], risk_free_rate=0, use_population=True)
    assert isinstance(result, float)

def test_sharpe_ratio_stdev_zero():
    with pytest.raises(ValueError):
        sharpe_ratio([2, 2, 2])

def test_sharpe_ratio_single_element_sample():
    with pytest.raises(ValueError):
        sharpe_ratio([5], use_population=False)
