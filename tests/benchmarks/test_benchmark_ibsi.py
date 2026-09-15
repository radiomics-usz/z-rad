import pytest

from .cases import pytest_params

pytestmark = pytest.mark.benchmark_ibsi


@pytest.mark.benchmark(group='ibsi1')
@pytest.mark.parametrize('case', pytest_params('test_ibsi_i_feature_workflow'))
def test_ibsi_i_feature_workflow(benchmark, measure, ibsi_feature_sources, case):
    measure(case.build(ibsi_feature_sources))


@pytest.mark.benchmark(group='ibsi2_phase1')
@pytest.mark.parametrize('case', pytest_params('test_ibsi_ii_filter_response'))
def test_ibsi_ii_filter_response(benchmark, measure, ibsi_ii_filter_sources, ibsi_ii_response_maps_dir, case):
    measure(case.build(ibsi_ii_filter_sources, ibsi_ii_response_maps_dir))


@pytest.mark.benchmark(group='ibsi2_phase2')
@pytest.mark.parametrize('case', pytest_params('test_ibsi_ii_feature_workflow'))
def test_ibsi_ii_feature_workflow(benchmark, measure, ibsi_ct_sources, case):
    measure(case.build(ibsi_ct_sources))
