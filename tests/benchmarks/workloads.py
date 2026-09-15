"""One workload definition for timing and memory; constructors perform setup.

Operations return new results. Input arrays are read-only to detect accidental
mutation. Filter metadata preparation is part of apply; output caching is absent.
"""

from dataclasses import dataclass
from functools import partial
from typing import Callable

import numpy as np

from zrad.filtering import Gabor, Laws, LoG, Mean, RieszLoG, Simoncelli, Wavelets2D, Wavelets3D
from zrad.image import Image
from zrad.preprocessing import (
    ImageResampler,
    IntensityMaskBuilder,
    IVHIntensityDiscretizer,
    MaskResampler,
    Resegmenter,
    RoiData,
    TextureDiscretizer,
)
from zrad.radiomics import Radiomics

IMAGE_SHAPES = {'small': (32, 96, 96), 'medium': (64, 128, 128), 'large': (96, 192, 192)}
ROI_SHAPES = {'small': (16, 24, 24), 'medium': (24, 40, 40), 'large': (40, 64, 64)}


@dataclass
class Workload:
    identifier: str
    operation: Callable
    validate: Callable
    metadata: dict
    rounds: int = 7
    setup: Callable | None = None


def image_from_array(array, spacing=(1.0, 1.0, 2.0)):
    return Image(
        array=array,
        origin=(0.0, 0.0, 0.0),
        spacing=spacing,
        direction=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        shape=array.shape[::-1],
    )


def synthetic_pair(shape):
    rng = np.random.default_rng(20260911)
    axes = np.ogrid[tuple(slice(0, size) for size in shape)]
    distance = sum(((axis - (size - 1) / 2) / (size * 0.38)) ** 2 for axis, size in zip(axes, shape))
    # Smooth anatomical-scale structure plus nonconstant texture; no huge bin range.
    array = np.rint(80 * np.cos(distance * 2) + rng.normal(0, 15, shape)).astype(np.float64)
    mask = (distance < 1).astype(np.float64)
    array.flags.writeable = mask.flags.writeable = False
    return image_from_array(array), image_from_array(mask)


def metadata(image, mask=None, **parameters):
    info = {
        'shape_zyx': list(image.array.shape),
        'voxel_count': int(image.array.size),
        'spacing_xyz_mm': list(image.spacing),
        'dtype': str(image.array.dtype),
        'dimensionality': '3D',
        'seed': 20260911,
        **parameters,
    }
    if mask is not None:
        info.update(roi_voxels=int(np.count_nonzero(mask.array)), roi_fraction=float(np.mean(mask.array > 0)))
    return info


def validate_image(result, expected_shape=None):
    assert isinstance(result, Image)
    assert result.array.size and np.isfinite(result.array).all()
    if expected_shape is not None:
        assert result.array.shape == expected_shape


def validate_features(result):
    assert result and all(np.isfinite(value) for value in result.values())


def prepare_roi(image, mask):
    roi = IntensityMaskBuilder().apply(RoiData(image=image, morphological_mask=mask))
    roi = Resegmenter(intensity_range=(-1000, 400)).apply(roi)
    roi = TextureDiscretizer(number_of_bins=32).apply(roi)
    return IVHIntensityDiscretizer(method='direct').apply(roi)


def resampling(size, method='Linear', target=False, mask=False, dimension='3D'):
    image, roi_mask = synthetic_pair(IMAGE_SHAPES[size])
    source = roi_mask if mask else image
    if dimension == '3D':
        resolution = (1.5, 1.5, 1.5)
    elif dimension == '2D':
        resolution = (1.5, 1.5, source.spacing[2])
    else:
        raise ValueError(f'Unsupported resample dimension {dimension!r}.')
    resampler = (MaskResampler if mask else ImageResampler)(resolution=resolution, method=method)
    if target:
        if dimension != '3D':
            raise ValueError('Target-grid benchmark uses the 3D resampling grid.')
        reference = resampler.apply(image)
        operation = partial(image.resample_to_target, reference)
        identifier = f'image/target_linear/{size}'
        shape = reference.array.shape
    else:
        operation = partial(resampler.apply, source)
        name = f'{"mask" if mask else "image"}_{method.lower()}'
        if dimension == '2D':
            name += '_in_plane'
        identifier = f'preprocessing/{name}/{size}'
        shape = tuple(np.ceil(np.array(source.shape) * np.array(source.spacing) / resolution).astype(int)[::-1])

    def validate(result):
        validate_image(result, shape)
        assert tuple(result.spacing) == resolution
        if mask:
            assert set(np.unique(result.array)) <= {0, 1}
            assert np.any(result.array > 0)
        if dimension == '2D':
            assert result.array.shape[0] == source.array.shape[0]

    return Workload(
        identifier,
        operation,
        validate,
        metadata(
            source,
            roi_mask,
            interpolation=method,
            resample_dimension=dimension,
            output_spacing_xyz_mm=list(resolution),
            output_shape_zyx=[int(axis) for axis in shape],
            output_voxel_count=int(np.prod(shape)),
        ),
        rounds=5 if size == 'large' else 7,
    )


def target_grid_alignment(size):
    image, roi_mask = synthetic_pair(IMAGE_SHAPES[size])
    spacing = (1.5, 1.5, 1.5)
    source_extent = np.array(image.shape) * np.array(image.spacing)
    target_shape = np.ceil(source_extent / spacing * (1.1, 0.9, 1.0)).astype(int)
    target_origin = tuple(source_extent * (0.15, -0.1, 0.2))
    target_array = np.zeros(tuple(target_shape[::-1]), dtype=np.float64)
    target_array.flags.writeable = False
    target = Image(
        array=target_array,
        origin=target_origin,
        spacing=spacing,
        direction=image.direction,
        shape=tuple(target_shape),
    )
    background = float(np.nanmin(image.array))

    def validate(result):
        validate_image(result, target_array.shape)
        assert tuple(result.origin) == target.origin
        assert tuple(result.spacing) == target.spacing
        assert tuple(result.direction) == target.direction
        assert tuple(result.shape) == target.shape
        assert result.array[0, 0, 0] == background
        assert result.array[-1, -1, -1] == background
        assert result.array[tuple(dimension // 2 for dimension in result.array.shape)] > background

    return Workload(
        f'image/target_partial_overlap_linear/{size}',
        partial(image.resample_to_target, target),
        validate,
        metadata(
            image,
            roi_mask,
            interpolation='Linear',
            target_shape_zyx=list(target_array.shape),
            target_origin_xyz_mm=list(target_origin),
            target_spacing_xyz_mm=list(spacing),
        ),
        rounds=5 if size == 'large' else 7,
    )


def preprocessing(operation):
    image, mask = synthetic_pair(IMAGE_SHAPES['medium'])
    roi = RoiData(image=image, morphological_mask=mask)
    step = IntensityMaskBuilder()
    if operation != 'roi':
        roi = step.apply(roi)
        step = (
            TextureDiscretizer(number_of_bins=32)
            if operation == 'discretize'
            else Resegmenter(intensity_range=(-50, 100), outlier_range=3)
        )

    def validate(result):
        values = (result.texture_discretized_image if operation == 'discretize' else result.intensity_mask).array
        assert np.isfinite(values).any() and not np.isinf(values).any()
        if operation == 'discretize':
            assert np.nanmin(values) == 1 and np.nanmax(values) == 32

    return Workload(
        f'preprocessing/{operation}/medium',
        partial(step.apply, roi),
        validate,
        metadata(image, mask, operation=operation, number_of_bins=32 if operation == 'discretize' else None),
    )


def texture_fixed_bin_size():
    image, mask = synthetic_pair(IMAGE_SHAPES['medium'])
    roi = IntensityMaskBuilder().apply(RoiData(image=image, morphological_mask=mask))
    roi = Resegmenter(intensity_range=(-50, 100)).apply(roi)
    step = TextureDiscretizer(bin_size=25)

    def validate(result):
        values = result.texture_discretized_image.array
        assert result.intensity_range == (-50.0, 100.0)
        assert np.isfinite(values).any() and not np.isinf(values).any()
        assert np.nanmin(values) == 1 and np.nanmax(values) == 7

    return Workload(
        'preprocessing/texture_fixed_bin_size_25/medium',
        partial(step.apply, roi),
        validate,
        metadata(
            image,
            mask,
            operation='texture_fixed_bin_size',
            bin_size=25,
            intensity_range=list(roi.intensity_range),
            prepared_roi_voxels=int(np.isfinite(roi.intensity_mask.array).sum()),
        ),
    )


def filtering(kind, size):
    image, _ = synthetic_pair(IMAGE_SHAPES[size])
    filters = {
        'mean_3d': lambda: Mean(padding_type='reflect', support=5, dimensionality='3D'),
        'mean_2d': lambda: Mean(padding_type='reflect', support=5, dimensionality='2D'),
        'log_3d': lambda: LoG(padding_type='reflect', sigma_mm=2.0, cutoff=4, dimensionality='3D'),
        'log_2d': lambda: LoG(padding_type='reflect', sigma_mm=2.0, cutoff=4, dimensionality='2D'),
        'wavelet_3d': lambda: Wavelets3D(
            wavelet_type='db3',
            padding_type='reflect',
            response_map='HHL',
            decomposition_level=1,
            rotation_invariance=False,
        ),
        'wavelet_3d_rot': lambda: Wavelets3D(
            wavelet_type='db3',
            padding_type='reflect',
            response_map='HHL',
            decomposition_level=1,
            rotation_invariance=True,
        ),
        'wavelet_2d_l1': lambda: Wavelets2D(
            wavelet_type='db3',
            padding_type='reflect',
            response_map='LH',
            decomposition_level=1,
            rotation_invariance=True,
        ),
        'wavelet_2d_l2': lambda: Wavelets2D(
            wavelet_type='db3',
            padding_type='reflect',
            response_map='LH',
            decomposition_level=2,
            rotation_invariance=False,
        ),
        'laws_plain': lambda: Laws(
            response_map='E3W5R5',
            padding_type='reflect',
            dimensionality='3D',
            rotation_invariance=False,
            pooling=None,
            energy_map=False,
            distance=7,
        ),
        'laws_rot_energy': lambda: Laws(
            response_map='E3W5R5',
            padding_type='reflect',
            dimensionality='3D',
            rotation_invariance=True,
            pooling='max',
            energy_map=True,
            distance=7,
        ),
        'gabor_fixed': lambda: Gabor(
            padding_type='reflect',
            res_mm=1.0,
            sigma_mm=5.0,
            lambda_mm=2.0,
            gamma=1.5,
            theta=np.pi / 8,
            rotation_invariance=False,
            orthogonal_planes=False,
        ),
        'gabor_rot_plane': lambda: Gabor(
            padding_type='reflect',
            res_mm=1.0,
            sigma_mm=5.0,
            lambda_mm=2.0,
            gamma=1.5,
            theta=np.pi / 8,
            rotation_invariance=True,
            orthogonal_planes=False,
        ),
        'gabor_rot_orthogonal': lambda: Gabor(
            padding_type='reflect',
            res_mm=1.0,
            sigma_mm=5.0,
            lambda_mm=2.0,
            gamma=1.5,
            theta=np.pi / 8,
            rotation_invariance=True,
            orthogonal_planes=True,
        ),
        'simoncelli_wrap_3d': lambda: Simoncelli(padding_type='periodic', decomposition_level=2, dimensionality='3D'),
        'simoncelli_nearest_3d': lambda: Simoncelli(padding_type='nearest', decomposition_level=2, dimensionality='3D'),
        'simoncelli_wrap_2d': lambda: Simoncelli(padding_type='periodic', decomposition_level=2, dimensionality='2D'),
        'simoncelli_riesz': lambda: Simoncelli(
            padding_type='periodic', decomposition_level=2, dimensionality='3D', riesz_order=(0, 2, 0)
        ),
        'riesz_first': lambda: RieszLoG(
            padding_type='reflect', sigma_mm=2.0, cutoff=4, dimensionality='3D', riesz_order=(1, 0, 0)
        ),
        'riesz_second': lambda: RieszLoG(
            padding_type='reflect', sigma_mm=2.0, cutoff=4, dimensionality='3D', riesz_order=(2, 0, 0)
        ),
        'riesz_aligned': lambda: RieszLoG(
            padding_type='reflect',
            sigma_mm=2.0,
            cutoff=4,
            dimensionality='3D',
            riesz_order=(2, 0, 0),
            structure_tensor_sigma_mm=1.0,
        ),
    }
    flt = filters[kind]()
    filter_params = flt.get_params()

    def validate(result):
        validate_image(result, image.array.shape)
        assert tuple(result.origin) == tuple(image.origin)
        assert tuple(result.spacing) == tuple(image.spacing)
        assert tuple(result.direction) == tuple(image.direction)
        assert tuple(result.shape) == tuple(image.shape)

    expensive = {
        'wavelet_3d_rot',
        'laws_rot_energy',
        'gabor_rot_plane',
        'gabor_rot_orthogonal',
        'simoncelli_nearest_3d',
        'riesz_first',
        'riesz_second',
        'riesz_aligned',
    }
    setup = flt._make_kernels.cache_clear if isinstance(flt, Gabor) else None
    extra = {}
    if isinstance(flt, Gabor):
        extra['gabor_kernel_state'] = 'fresh_each_round'
    if kind == 'wavelet_2d_l2':
        extra['effective_rotation_count'] = 4
    rounds = (
        3 if kind in {'laws_rot_energy', 'gabor_rot_orthogonal', 'riesz_aligned'} else (5 if kind in expensive else 7)
    )
    return Workload(
        f'filtering/{kind}/{size}',
        partial(flt.apply, image),
        validate,
        metadata(
            image,
            dimensionality=filter_params.get('dimensionality', '2D planes'),
            filter=filter_params,
            **extra,
        ),
        rounds=rounds,
        setup=setup,
    )


def clear_radiomics_result_caches():
    """Per-round setup only: exclude retained image results from extraction."""
    from zrad.radiomics.intensity import _LOCAL_MEANS_CACHE

    # This is the only module-level radiomics result cache. New contexts and
    # feature groups alone do not reset it because its key includes array identity.
    _LOCAL_MEANS_CACHE.clear()


def radiomics(size, family='all', aggregation=('3D', 'MERG')):
    image, mask = synthetic_pair(ROI_SHAPES[size])
    roi = prepare_roi(image, mask)
    for value in vars(roi).values():
        if isinstance(value, Image):
            value.array.flags.writeable = False
    extractor = Radiomics(aggr_dim=aggregation[0], aggr_method=aggregation[1])
    if family == 'spatial':
        operation = partial(extractor.extract_features, roi_data=roi, features=['morph_moran_i', 'morph_geary_c'])
    else:
        families = ['glcm', 'glrlm', 'glszm', 'gldzm', 'ngtdm', 'ngldm'] if family == 'texture' else family
        operation = partial(extractor.extract_features, roi_data=roi, families=families)

    def validate(result):
        validate_features(result)
        if family == 'all':
            assert len(result) > 150 and 'morph_moran_i' in result and 'ivh_v10' in result
        if family == 'spatial':
            assert set(result) == {'morph_moran_i', 'morph_geary_c'}

    family_id = 'all_fresh' if family == 'all' else family
    aggregation_id = f'/{aggregation[0].lower().replace(".", "_")}/{aggregation[1].lower()}'
    if aggregation == ('3D', 'MERG'):
        aggregation_id = ''
    return Workload(
        f'radiomics/{family_id}/{size}{aggregation_id}',
        operation,
        validate,
        metadata(
            image,
            mask,
            feature_group=family,
            number_of_bins=32,
            aggregation='/'.join(aggregation),
            **({'cache_policy': 'fresh_image_results'} if family == 'all' else {}),
        ),
        rounds=5,
        setup=clear_radiomics_result_caches if family in ('all', 'local_intensity') else None,
    )
