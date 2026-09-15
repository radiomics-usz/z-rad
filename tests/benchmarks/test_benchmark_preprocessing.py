import pytest

from .cases import pytest_params, single_case

pytestmark = pytest.mark.benchmark(group='preprocessing')


@pytest.mark.parametrize('case', pytest_params('test_image_resampling_isotropic'))
def test_image_resampling_isotropic(benchmark, measure, case):
    measure(case.build())


@pytest.mark.parametrize('case', pytest_params('test_mask_resampling_isotropic'))
def test_mask_resampling_isotropic(benchmark, measure, case):
    measure(case.build())


@pytest.mark.parametrize('case', pytest_params('test_image_resampling_in_plane'))
def test_image_resampling_in_plane(benchmark, measure, case):
    measure(case.build())


@pytest.mark.parametrize('case', pytest_params('test_mask_resampling_in_plane'))
def test_mask_resampling_in_plane(benchmark, measure, case):
    measure(case.build())


def test_intensity_mask_building(benchmark, measure):
    measure(single_case('test_intensity_mask_building').build())


def test_range_and_outlier_resegmentation(benchmark, measure):
    measure(single_case('test_range_and_outlier_resegmentation').build())


def test_texture_discretization_fixed_bin_number(benchmark, measure):
    measure(single_case('test_texture_discretization_fixed_bin_number').build())


def test_texture_discretization_fixed_bin_size(benchmark, measure):
    measure(single_case('test_texture_discretization_fixed_bin_size').build())
