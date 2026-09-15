import pytest

from .cases import pytest_params

pytestmark = pytest.mark.benchmark(group='image')


@pytest.mark.parametrize('case', pytest_params('test_target_resampling'))
def test_target_resampling(benchmark, measure, case):
    measure(case.build())


@pytest.mark.parametrize('case', pytest_params('test_target_grid_alignment'))
def test_target_grid_alignment(benchmark, measure, case):
    measure(case.build())
