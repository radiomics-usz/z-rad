import pytest

from .cases import pytest_params

pytestmark = pytest.mark.benchmark(group='radiomics')


@pytest.mark.parametrize('case', pytest_params('test_complete_fresh_extraction'))
def test_complete_fresh_extraction(benchmark, measure, case):
    measure(case.build())


@pytest.mark.parametrize('case', pytest_params('test_spatial_statistics'))
def test_spatial_statistics(benchmark, measure, case):
    measure(case.build())


@pytest.mark.parametrize('case', pytest_params('test_feature_family'))
def test_feature_family(benchmark, measure, case):
    measure(case.build())


@pytest.mark.parametrize('case', pytest_params('test_texture_aggregation'))
def test_texture_aggregation(benchmark, measure, case):
    measure(case.build())
