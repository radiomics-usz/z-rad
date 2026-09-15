"""Canonical case inventory shared by timing collection and isolated memory runs.

Case declarations are cheap to import; inputs and numerical workloads are built
only when a selected case executes. IDs describe the operation and its parameters.
"""

from dataclasses import dataclass

from ibsi_cases import IBSI_I_FEATURE_CASES, IBSI_II_FEATURE_CASES, IBSI_II_FILTER_CASES


@dataclass(frozen=True)
class Case:
    identifier: str
    test: str
    pytest_id: str
    factory: str
    args: tuple = ()
    kwargs: tuple = ()
    group: str = ''
    exhaustive: bool = False
    slow: bool = False
    reference: object = None

    def build(self, sources=None, response_maps=None):
        """Construct the same workload for either measurement backend."""
        if self.group == 'ibsi2_phase1':
            from .ibsi_workloads import ibsi_filter

            workload = ibsi_filter(self.reference, sources, response_maps)
        elif self.group in ('ibsi1', 'ibsi2_phase2'):
            from .ibsi_workloads import ibsi_feature

            workload = ibsi_feature(self.reference, sources)
        else:
            from . import workloads

            workload = getattr(workloads, self.factory)(*self.args, **dict(self.kwargs))
        if workload.identifier != self.identifier:
            raise RuntimeError(f'Case ID {self.identifier!r} built workload {workload.identifier!r}.')
        return workload


def synthetic(identifier, test, pytest_id, factory, *args, group, kwargs=(), slow=False):
    return Case(identifier, test, pytest_id, factory, args, kwargs, group, slow=slow)


IMAGE_CASES = tuple(
    synthetic(
        f'image/target_linear/{size}',
        'test_target_resampling',
        size,
        'resampling',
        size,
        group='image',
        kwargs=(('target', True),),
        slow=size == 'large',
    )
    for size in ('small', 'medium', 'large')
) + tuple(
    synthetic(
        f'image/target_partial_overlap_linear/{size}',
        'test_target_grid_alignment',
        size,
        'target_grid_alignment',
        size,
        group='image',
        slow=size == 'large',
    )
    for size in ('small', 'medium', 'large')
)

PREPROCESSING_CASES = (
    tuple(
        synthetic(
            f'preprocessing/image_{method.lower()}/{size}',
            'test_image_resampling_isotropic',
            f'{"nearest_neighbor" if method == "NN" else method.lower()}-{size}',
            'resampling',
            size,
            method,
            group='preprocessing',
            slow=size == 'large',
        )
        for method, sizes in (
            ('Linear', ('small', 'medium', 'large')),
            ('BSpline', ('small', 'medium', 'large')),
            ('NN', ('medium',)),
            ('Gaussian', ('medium',)),
        )
        for size in sizes
    )
    + tuple(
        synthetic(
            f'preprocessing/mask_{method.lower()}/medium',
            'test_mask_resampling_isotropic',
            f'{"nearest_neighbor" if method == "NN" else method.lower()}-medium',
            'resampling',
            'medium',
            method,
            group='preprocessing',
            kwargs=(('mask', True),),
        )
        for method in ('NN', 'Linear', 'BSpline', 'Gaussian')
    )
    + (
        synthetic(
            'preprocessing/image_linear_in_plane/medium',
            'test_image_resampling_in_plane',
            'linear-medium',
            'resampling',
            'medium',
            'Linear',
            group='preprocessing',
            kwargs=(('dimension', '2D'),),
        ),
    )
    + tuple(
        synthetic(
            f'preprocessing/mask_{method.lower()}_in_plane/medium',
            'test_mask_resampling_in_plane',
            f'{"nearest_neighbor" if method == "NN" else method.lower()}-medium',
            'resampling',
            'medium',
            method,
            group='preprocessing',
            kwargs=(('mask', True), ('dimension', '2D')),
        )
        for method in ('NN', 'Linear')
    )
    + tuple(
        synthetic(f'preprocessing/{operation}/medium', test, '', 'preprocessing', operation, group='preprocessing')
        for operation, test in (
            ('roi', 'test_intensity_mask_building'),
            ('resegment', 'test_range_and_outlier_resegmentation'),
            ('discretize', 'test_texture_discretization_fixed_bin_number'),
        )
    )
    + (
        synthetic(
            'preprocessing/texture_fixed_bin_size_25/medium',
            'test_texture_discretization_fixed_bin_size',
            '',
            'texture_fixed_bin_size',
            group='preprocessing',
        ),
    )
)

FILTER_PARAMETERS = (
    ('mean_3d', 'medium'),
    ('mean_2d', 'medium'),
    ('log_3d', 'medium'),
    ('log_2d', 'medium'),
    ('wavelet_3d', 'medium'),
    ('wavelet_3d_rot', 'medium'),
    ('wavelet_2d_l1', 'small'),
    ('wavelet_2d_l2', 'small'),
    ('laws_plain', 'medium'),
    ('laws_rot_energy', 'medium'),
    ('gabor_fixed', 'small'),
    ('gabor_rot_plane', 'small'),
    ('gabor_rot_orthogonal', 'small'),
    ('simoncelli_wrap_3d', 'medium'),
    ('simoncelli_nearest_3d', 'medium'),
    ('simoncelli_wrap_2d', 'medium'),
    ('simoncelli_riesz', 'medium'),
    ('riesz_first', 'medium'),
    ('riesz_second', 'small'),
    ('riesz_aligned', 'small'),
)
FILTER_SLOW = {
    'wavelet_3d_rot',
    'wavelet_2d_l1',
    'wavelet_2d_l2',
    'laws_rot_energy',
    'gabor_rot_plane',
    'gabor_rot_orthogonal',
    'simoncelli_nearest_3d',
    'simoncelli_riesz',
    'riesz_first',
    'riesz_second',
    'riesz_aligned',
}
FILTERING_CASES = tuple(
    synthetic(
        f'filtering/{kind}/{size}',
        'test_apply',
        f'{kind}-{size}',
        'filtering',
        kind,
        size,
        group='filtering',
        slow=kind in FILTER_SLOW,
    )
    for kind, size in FILTER_PARAMETERS
)

FEATURE_FAMILIES = (
    'morphology',
    'local_intensity',
    'intensity_statistics',
    'intensity_histogram',
    'glcm',
    'glrlm',
    'glszm',
    'gldzm',
    'ngtdm',
    'ngldm',
    'ivh',
)
DIRECTIONAL_AGGREGATIONS = (
    ('2D', 'AVER'),
    ('2D', 'SLICE_MERG'),
    ('2.5D', 'DIR_MERG'),
    ('2.5D', 'MERG'),
    ('3D', 'AVER'),
)
DIMENSIONAL_AGGREGATIONS = (('2D', 'AVER'), ('2.5D', 'AVER'), ('3D', 'AVER'))
AGGREGATION_CASES = tuple(
    (family, aggregation) for family in ('glcm', 'glrlm') for aggregation in DIRECTIONAL_AGGREGATIONS
) + tuple(
    (family, aggregation) for family in ('glszm', 'gldzm', 'ngtdm', 'ngldm') for aggregation in DIMENSIONAL_AGGREGATIONS
)

RADIOMICS_CASES = (
    (
        synthetic(
            'radiomics/all_fresh/medium',
            'test_complete_fresh_extraction',
            'medium',
            'radiomics',
            'medium',
            group='radiomics',
        ),
        synthetic(
            'radiomics/spatial/medium',
            'test_spatial_statistics',
            'medium',
            'radiomics',
            'medium',
            'spatial',
            group='radiomics',
        ),
    )
    + tuple(
        synthetic(
            f'radiomics/{family}/medium',
            'test_feature_family',
            family,
            'radiomics',
            'medium',
            family,
            group='radiomics',
        )
        for family in FEATURE_FAMILIES
    )
    + tuple(
        synthetic(
            f'radiomics/{family}/medium/{aggregation[0].lower().replace(".", "_")}/{aggregation[1].lower()}',
            'test_texture_aggregation',
            f'{family}-{aggregation[0]}/{aggregation[1]}',
            'radiomics',
            'medium',
            family,
            aggregation,
            group='radiomics',
            slow=True,
        )
        for family, aggregation in AGGREGATION_CASES
    )
)

IBSI_CASES = (
    tuple(
        Case(
            case.identifier,
            'test_ibsi_i_feature_workflow',
            case.identifier,
            'ibsi_feature',
            group='ibsi1',
            slow=True,
            reference=case,
        )
        for case in IBSI_I_FEATURE_CASES
    )
    + tuple(
        Case(
            case.identifier,
            'test_ibsi_ii_filter_response',
            case.identifier,
            'ibsi_filter',
            group='ibsi2_phase1',
            exhaustive=True,
            slow=True,
            reference=case,
        )
        for case in IBSI_II_FILTER_CASES
    )
    + tuple(
        Case(
            case.identifier,
            'test_ibsi_ii_feature_workflow',
            case.identifier,
            'ibsi_feature',
            group='ibsi2_phase2',
            exhaustive=True,
            slow=True,
            reference=case,
        )
        for case in IBSI_II_FEATURE_CASES
    )
)

ALL_CASES = IMAGE_CASES + PREPROCESSING_CASES + FILTERING_CASES + RADIOMICS_CASES + IBSI_CASES
CASE_BY_ID = {case.identifier: case for case in ALL_CASES}
if len(ALL_CASES) != len(CASE_BY_ID):
    raise RuntimeError('Duplicate benchmark workload IDs in case registry.')


def suite_cases(name):
    if name == 'standard':
        return tuple(case for case in ALL_CASES if not case.exhaustive)
    if name == 'ibsi':
        return IBSI_CASES
    if name == 'exhaustive':
        return ALL_CASES
    raise ValueError(f'Unknown benchmark suite {name!r}.')


def pytest_params(test):
    """Apply the collection IDs and slow/exhaustive markers from this inventory."""
    import pytest

    return tuple(
        pytest.param(
            case,
            id=case.pytest_id or None,
            marks=[
                mark
                for enabled, mark in (
                    (case.slow, pytest.mark.benchmark_slow),
                    (case.exhaustive, pytest.mark.benchmark_exhaustive),
                )
                if enabled
            ],
        )
        for case in ALL_CASES
        if case.test == test
    )


def single_case(test):
    matches = tuple(case for case in ALL_CASES if case.test == test)
    if len(matches) != 1:
        raise RuntimeError(f'Expected one benchmark case for {test!r}; got {len(matches)}.')
    return matches[0]
