"""Shared, declarative inventory of published IBSI benchmark cases.

The correctness and performance suites consume these definitions so adding or
changing a published configuration cannot silently update only one suite.
"""

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping


@dataclass(frozen=True)
class IbsiFeatureCase:
    identifier: str
    phase: str
    config: str
    image_source: str
    mask_source: str
    resampling_dim: str | None
    image_interpolation: str | None
    preparation: Mapping
    aggregation: tuple[str, str]
    filter_method: str | None = None
    filter_params: Mapping | None = None


@dataclass(frozen=True)
class IbsiFilterCase:
    identifier: str
    config: str
    phantom: str
    response_map: str
    filter_method: str
    filter_params: Mapping


def _frozen(values=None):
    return MappingProxyType(dict(values or {}))


def _feature_case(
    identifier,
    phase,
    config,
    image_source,
    mask_source,
    resampling_dim,
    image_interpolation,
    preparation,
    aggregation,
    filter_method=None,
    filter_params=None,
):
    return IbsiFeatureCase(
        identifier=identifier,
        phase=phase,
        config=config,
        image_source=image_source,
        mask_source=mask_source,
        resampling_dim=resampling_dim,
        image_interpolation=image_interpolation,
        preparation=_frozen(preparation),
        aggregation=aggregation,
        filter_method=filter_method,
        filter_params=_frozen(filter_params) if filter_params is not None else None,
    )


def _filter_case(config, phantom, response_filename, method, **params):
    return IbsiFilterCase(
        identifier=f'ibsi/ii/phase_i/{config}',
        config=config,
        phantom=phantom,
        response_map=response_filename,
        filter_method=method,
        filter_params=_frozen(params),
    )


_IBSI_I_AGGREGATIONS = {
    'A': (('2D', 'AVER'), ('2D', 'SLICE_MERG'), ('2.5D', 'DIR_MERG'), ('2.5D', 'MERG')),
    'B': (('2D', 'AVER'), ('2D', 'SLICE_MERG'), ('2.5D', 'DIR_MERG'), ('2.5D', 'MERG')),
    'C': (('3D', 'AVER'), ('3D', 'MERG')),
    'D': (('3D', 'AVER'), ('3D', 'MERG')),
    'E': (('3D', 'AVER'), ('3D', 'MERG')),
    'digital': (
        ('2D', 'AVER'),
        ('2D', 'SLICE_MERG'),
        ('2.5D', 'DIR_MERG'),
        ('2.5D', 'MERG'),
        ('3D', 'AVER'),
        ('3D', 'MERG'),
    ),
}

_IBSI_I_CONFIGS = {
    'A': dict(
        image_source='ct_dicom',
        mask_source='ct_dicom',
        resampling_dim=None,
        image_interpolation=None,
        preparation=dict(intensity_range=(-500, 400), bin_size=25, ivh_method='direct'),
    ),
    'B': dict(
        image_source='ct_nifti',
        mask_source='ct_nifti',
        resampling_dim='2D',
        image_interpolation='Linear',
        preparation=dict(intensity_range=(-500, 400), number_of_bins=32, ivh_method='direct'),
    ),
    'C': dict(
        image_source='ct_nifti',
        mask_source='ct_nifti',
        resampling_dim='3D',
        image_interpolation='Linear',
        preparation=dict(intensity_range=(-1000, 400), bin_size=25, ivh_method='fixed_bin_size', ivh_bin_size=2.5),
    ),
    'D': dict(
        image_source='ct_nifti',
        mask_source='ct_nifti',
        resampling_dim='3D',
        image_interpolation='Linear',
        preparation=dict(outlier_range=3, number_of_bins=32, ivh_method='direct'),
    ),
    'E': dict(
        image_source='ct_dicom',
        mask_source='ct_nifti',
        resampling_dim='3D',
        image_interpolation='BSpline',
        preparation=dict(
            intensity_range=(-1000, 400),
            outlier_range=3,
            number_of_bins=32,
            ivh_method='fixed_bin_number',
            ivh_number_of_bins=1000,
        ),
    ),
    'digital': dict(
        image_source='i_digital',
        mask_source='i_digital',
        resampling_dim=None,
        image_interpolation=None,
        preparation=dict(number_of_bins=6, ivh_method='direct'),
    ),
}

IBSI_I_FEATURE_CASES = tuple(
    _feature_case(
        identifier=f'ibsi/i/{config.lower()}/{dimension.lower().replace(".", "_")}/{method.lower()}',
        phase='I',
        config=config,
        aggregation=(dimension, method),
        **settings,
    )
    for config, settings in _IBSI_I_CONFIGS.items()
    for dimension, method in _IBSI_I_AGGREGATIONS[config]
)


IBSI_II_FILTER_CASES = (
    *(
        _filter_case(config, phantom, filename, 'Mean', padding_type=padding, dimensionality=dimension, support=15)
        for config, padding, dimension, phantom, filename in (
            ('1.a.1', 'constant', '3D', 'checkerboard', '1_a_1-ValidCRM.nii'),
            ('1.a.2', 'nearest', '3D', 'checkerboard', '1_a_2-ValidCRM.nii'),
            ('1.a.3', 'wrap', '3D', 'checkerboard', '1_a_3-ValidCRM.nii'),
            ('1.a.4', 'reflect', '3D', 'checkerboard', '1_a_4-ValidCRM.nii'),
            ('1.b.1', 'constant', '2D', 'impulse', '1_b_1-ValidCRM.nii'),
        )
    ),
    *(
        _filter_case(
            config,
            phantom,
            filename,
            'Laplacian of Gaussian',
            padding_type=padding,
            dimensionality=dimension,
            sigma_mm=sigma,
            cutoff=4,
        )
        for config, padding, dimension, sigma, phantom, filename in (
            ('2.a', 'constant', '3D', 3.0, 'impulse', '2_a-ValidCRM.nii'),
            ('2.b', 'reflect', '3D', 5.0, 'checkerboard', '2_b-ValidCRM.nii'),
            ('2.c', 'reflect', '2D', 5.0, 'checkerboard', '2_c-ValidCRM.nii'),
        )
    ),
    *(
        _filter_case(
            config,
            phantom,
            filename,
            'Laws Kernels',
            padding_type=padding,
            dimensionality=dimension,
            response_map=response,
            rotation_invariance=rotation,
            pooling=pooling,
            energy_map=energy,
            distance=distance,
        )
        for config, padding, dimension, response, rotation, pooling, energy, distance, phantom, filename in (
            ('3.a.1', 'constant', '3D', 'E5L5S5', False, None, False, 0, 'impulse', '3_a_1-ValidCRM.nii'),
            ('3.a.2', 'constant', '3D', 'E5L5S5', True, 'max', False, 0, 'impulse', '3_a_2-ValidCRM.nii'),
            ('3.a.3', 'constant', '3D', 'E5L5S5', True, 'max', True, 7, 'impulse', '3_a_3-ValidCRM.nii'),
            ('3.b.1', 'reflect', '3D', 'E3W5R5', False, None, False, 0, 'checkerboard', '3_b_1-ValidCRM.nii'),
            ('3.b.2', 'reflect', '3D', 'E3W5R5', True, 'max', False, 0, 'checkerboard', '3_b_2-ValidCRM.nii'),
            ('3.b.3', 'reflect', '3D', 'E3W5R5', True, 'max', True, 7, 'checkerboard', '3_b_3-ValidCRM.nii'),
            ('3.c.1', 'reflect', '2D', 'L5S5', False, None, False, 0, 'checkerboard', '3_c_1-ValidCRM.nii'),
            ('3.c.2', 'reflect', '2D', 'L5S5', True, 'max', False, 0, 'checkerboard', '3_c_2-ValidCRM.nii'),
            ('3.c.3', 'reflect', '2D', 'L5S5', True, 'max', True, 7, 'checkerboard', '3_c_3-ValidCRM.nii'),
        )
    ),
    *(
        _filter_case(
            config,
            phantom,
            filename,
            'Gabor',
            padding_type=padding,
            res_mm=2.0,
            sigma_mm=sigma,
            lambda_mm=wave_length,
            gamma=gamma,
            theta=theta,
            rotation_invariance=rotation,
            orthogonal_planes=orthogonal,
            n_stds=n_stds,
        )
        for config, padding, sigma, wave_length, gamma, theta, rotation, orthogonal, n_stds, phantom, filename in (
            ('4.a.1', 'constant', 10.0, 4.0, 0.5, math.pi / 3, False, False, 11, 'impulse', '4_a_1-ValidCRM.nii'),
            ('4.a.2', 'constant', 10.0, 4.0, 0.5, math.pi / 4, True, True, 11, 'impulse', '4_a_2-ValidCRM.nii'),
            ('4.b.1', 'reflect', 20.0, 8.0, 2.5, 5 * math.pi / 4, False, False, None, 'sphere', '4_b_1-ValidCRM.nii'),
            ('4.b.2', 'reflect', 20.0, 8.0, 2.5, math.pi / 8, True, True, None, 'sphere', '4_b_2-ValidCRM.nii'),
        )
    ),
    *(
        _filter_case(
            config,
            phantom,
            filename,
            'Wavelets',
            wavelet_type=wavelet,
            dimensionality='3D',
            padding_type=padding,
            response_map=response,
            decomposition_level=level,
            rotation_invariance=rotation,
        )
        for config, wavelet, padding, response, level, rotation, phantom, filename in (
            ('5.a.1', 'db2', 'constant', 'LHL', 1, False, 'impulse', '5_a_1-ValidCRM.nii'),
            ('5.a.2', 'db2', 'constant', 'LHL', 1, True, 'impulse', '5_a_2-ValidCRM.nii'),
            ('6.a.1', 'coif1', 'wrap', 'HHL', 1, False, 'sphere', '6_a_1-ValidCRM.nii'),
            ('6.a.2', 'coif1', 'wrap', 'HHL', 1, True, 'sphere', '6_a_2-ValidCRM.nii'),
            ('7.a.1', 'haar', 'reflect', 'LLL', 2, False, 'checkerboard', '7_a_1-ValidCRM.nii'),
            ('7.a.2', 'haar', 'reflect', 'HHH', 2, True, 'checkerboard', '7_a_2-ValidCRM.nii'),
        )
    ),
    *(
        _filter_case(
            f'8.a.{level}',
            'checkerboard',
            f'8_a_{level}-ValidCRM.nii',
            'Simoncelli',
            dimensionality='3D',
            padding_type='wrap',
            decomposition_level=level,
        )
        for level in (1, 2, 3)
    ),
    *(
        _filter_case(
            config,
            phantom,
            filename,
            'Riesz-transformed LoG',
            dimensionality='3D',
            padding_type='constant',
            sigma_mm=3.0,
            cutoff=4,
            riesz_order=order,
        )
        for config, phantom, order, filename in (
            ('9.a', 'impulse', (1, 0, 0), '9_a-ValidCRM.nii'),
            ('9.b.1', 'sphere', (0, 2, 0), '9_b_1-ValidCRM.nii'),
        )
    ),
    _filter_case(
        '10.b.1',
        'pattern_1',
        '10_b_1-ValidCRM.nii',
        'Simoncelli',
        dimensionality='3D',
        padding_type='nearest',
        decomposition_level=1,
        riesz_order=(0, 2, 0),
    ),
)


_IBSI_II_PHASE_II_FILTERS = {
    '1': (None, None),
    '2': ('Mean', dict(padding_type='reflect', support=5)),
    '3': ('Laplacian of Gaussian', dict(padding_type='reflect', sigma_mm=1.5, cutoff=4)),
    '4': (
        'Laws Kernels',
        dict(padding_type='reflect', rotation_invariance=True, pooling='max', energy_map=True, distance=7),
    ),
    '5': (
        'Gabor',
        dict(
            padding_type='reflect',
            sigma_mm=5.0,
            lambda_mm=2.0,
            gamma=1.5,
            theta=math.pi / 8,
            rotation_invariance=True,
        ),
    ),
    '6': (
        'Wavelets',
        dict(wavelet_type='db3', padding_type='reflect', decomposition_level=1, rotation_invariance=True),
    ),
    '7': (
        'Wavelets',
        dict(wavelet_type='db3', padding_type='reflect', decomposition_level=2, rotation_invariance=True),
    ),
    '8': ('Simoncelli', dict(padding_type='periodic', decomposition_level=1)),
    '9': ('Simoncelli', dict(padding_type='periodic', decomposition_level=2)),
}


def _phase_ii_filter(config):
    number, variant = config.split('.')
    method, common = _IBSI_II_PHASE_II_FILTERS[number]
    if method is None:
        return None, None
    dimension = '2D' if variant == 'A' else '3D'
    params = dict(common)
    params['dimensionality'] = dimension
    if number == '4':
        params['response_map'] = 'L5E5' if variant == 'A' else 'L5E5E5'
    elif number == '5':
        params.update(res_mm=0.977 if variant == 'A' else 1.0, orthogonal_planes=variant == 'B')
    elif number == '6':
        params['response_map'] = 'LH' if variant == 'A' else 'LLH'
    elif number == '7':
        params['response_map'] = 'HH' if variant == 'A' else 'HHH'
    return method, params


IBSI_II_FEATURE_CASES = tuple(
    _feature_case(
        identifier=f'ibsi/ii/phase_ii/{config.lower()}',
        phase='II',
        config=config,
        image_source='ct_dicom',
        mask_source='ct_dicom',
        resampling_dim='3D' if config.endswith('.B') else None,
        image_interpolation='BSpline' if config.endswith('.B') else None,
        preparation=dict(intensity_range=(-1000, 400), bin_size=25),
        aggregation=('2D', 'AVER'),
        filter_method=_phase_ii_filter(config)[0],
        filter_params=_phase_ii_filter(config)[1],
    )
    for number in range(1, 10)
    for config in (f'{number}.A', f'{number}.B')
)


ALL_IBSI_PERFORMANCE_CASES = IBSI_I_FEATURE_CASES + IBSI_II_FILTER_CASES + IBSI_II_FEATURE_CASES
