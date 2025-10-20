import pytest 
from quantmath import *


def test_arithmetic_mean():
    data = [1,2,3]
    assert arithmetic_mean(data) == 2.0

def test_weighted_mean():
    weights = [10,20,30]
    data = [1,2,3]
    assert weighted_mean(data, weights) == pytest.approx(2.408)


def test_geometric_mean():
    data = [1,2,4]
    assert geometric_mean(data) == 2.0


def test_population_variance():
    data = [1,2,3]
    assert population_variance(data) == 2/3


def test_sharpe_ratio():
    returns = [0.01, 0.005, -0.002, 0.007, 0.003]
    use_population = False
    risk_free = 0.001     
    assert sharpe_ratio(returns, risk_free, use_population) == pytest.approx(1.54)