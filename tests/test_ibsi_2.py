from pathlib import Path

import numpy as np
import pytest
from ibsi_cases import IBSI_II_FEATURE_CASES, IBSI_II_FILTER_CASES
from ibsi_helpers import load_references, matches_reference

from zrad.filtering import create_filter
from zrad.image import Image
from zrad.preprocessing import (
    ImageResampler,
    IntensityMaskBuilder,
    MaskResampler,
    Resegmenter,
    RoiData,
    TextureDiscretizer,
)
from zrad.radiomics import Radiomics


def _phase_i_cases(prefix):
    return tuple(case for case in IBSI_II_FILTER_CASES if case.config.startswith(prefix))


def _phase_i_case(config):
    return next(case for case in IBSI_II_FILTER_CASES if case.config == config)


def _phase_ii_case(config):
    return next(case for case in IBSI_II_FEATURE_CASES if case.config == config)


def _phase_ii_filter(config):
    case = _phase_ii_case(config)
    assert case.filter_method is not None
    return create_filter(filtering_method=case.filter_method, **dict(case.filter_params))


def _run_ph_i_case(filtering, phantom, filename, config, data_dir):
    filtered_image = filtering.apply(phantom)
    response_map_path = data_dir / 'reference_response_maps' / filename
    response_map = Image.from_nifti(str(response_map_path))
    ibsi_ii_ph_i_validation(filtered_image.array, response_map.array, config)


def ibsi_ii_feature_tolerances(filter_id):
    csv_path = (
        Path(__file__).parent / 'data' / 'ibsi_2_reference_data' / 'reference_feature_values' / 'reference_values.csv'
    )
    return load_references(csv_path, 'feature_tag', 'consensus_value', delimiter=';', phase='II', config=filter_id)


def ibsi_ii_ph_i_validation(filtered_image, response_map, config_id):

    assert filtered_image.shape == response_map.shape, f"{config_id}: response-map shape mismatch"
    assert response_map.size > 0, f"{config_id}: empty response map"
    assert np.isfinite(response_map).all(), f"{config_id}: non-finite reference voxels"
    assert np.isfinite(filtered_image).all(), f"{config_id}: non-finite computed voxels"
    tolerance = 0.01 * np.ptp(response_map)
    errors = np.abs(filtered_image - response_map)
    failing = np.count_nonzero(errors > tolerance)
    if failing:
        pytest.fail(
            f"{config_id}: {failing}/{response_map.size} voxels out of tolerance; "
            f"maximum error={errors.max():.8g}, tolerance={tolerance:.8g}"
        )


def ibsi_ii_ph_ii_validation(ibsi_features, features, config_8b=False):

    assert ibsi_features, "Empty IBSI reference selection"
    for raw_tag, feature_info in ibsi_features.items():
        tag = str(raw_tag)
        if config_8b and tag == 'stat_qcod':
            # IBSI II reference manual, Table 7.16: consensus is "none" for
            # 8.B stat_qcod, so IBSI publishes no reference value or tolerance.
            # https://doi.org/10.48550/arXiv.2006.05470
            assert feature_info.get('filter_id') == '8.B', '8.B exception used for another configuration'
            assert feature_info['consensus_value'] == feature_info['tolerance'] == '', (
                'Reference availability changed; review 8.B exception'
            )
            continue

        if tag not in features:
            pytest.fail(f"Missing required feature {tag}")

        if not matches_reference(features[tag], feature_info['consensus_value'], feature_info['tolerance']):
            pytest.fail(
                f"Feature {tag} out of tolerance: computed={features[tag]}, "
                f"reference={feature_info['consensus_value']}, tolerance={feature_info['tolerance']}"
            )


@pytest.mark.unit
@pytest.mark.parametrize('config_8b', [False, True])
@pytest.mark.parametrize('features', [{}, {'stat_mean': 1.0}])
def test_ibsi_ii_requires_all_reference_features(config_8b, features):
    reference = {
        'stat_mean': {'consensus_value': '1', 'tolerance': '0'},
        'stat_var': {'consensus_value': '2', 'tolerance': '0'},
    }
    missing = 'stat_var' if features else 'stat_mean'
    with pytest.raises(pytest.fail.Exception, match=f'Missing required feature {missing}'):
        ibsi_ii_ph_ii_validation(reference, features, config_8b=config_8b)


@pytest.mark.unit
def test_ibsi_ii_8b_excludes_only_qcod():
    reference = ibsi_ii_feature_tolerances('8.B')
    features = {tag: float(info['consensus_value']) for tag, info in reference.items() if tag != 'stat_qcod'}
    ibsi_ii_ph_ii_validation(reference, features, config_8b=True)
    del features['stat_mean']
    with pytest.raises(pytest.fail.Exception, match='Missing required feature stat_mean'):
        ibsi_ii_ph_ii_validation(reference, features, config_8b=True)


@pytest.mark.unit
def test_ibsi_ii_requires_qcod_outside_8b():
    reference = {'stat_qcod': ibsi_ii_feature_tolerances('8.A')['stat_qcod']}
    with pytest.raises(pytest.fail.Exception, match='Missing required feature stat_qcod'):
        ibsi_ii_ph_ii_validation(reference, {})


def _extract_filtered_features(image, filtered_image, mask, aggr_dim='2D', aggr_method='AVER'):
    roi_data = IntensityMaskBuilder().apply(
        RoiData(
            image=image,
            filtered_image=filtered_image,
            morphological_mask=mask,
        )
    )
    roi_data = Resegmenter(intensity_range=[-1000, 400]).apply(roi_data)
    roi_data = TextureDiscretizer(bin_size=25).apply(roi_data)
    return Radiomics(
        aggr_dim=aggr_dim,
        aggr_method=aggr_method,
    ).extract_features(roi_data=roi_data)


@pytest.fixture()
def ct_phantom_image(ibsi_ct_data_dir):
    return Image.from_dicom(dicom_dir=str(ibsi_ct_data_dir / 'dicom' / 'image'), modality='CT')


@pytest.fixture()
def ct_phantom_mask(ct_phantom_image, ibsi_ct_data_dir):
    return Image.from_dicom_mask(
        rtstruct_path=str(ibsi_ct_data_dir / 'dicom' / 'mask' / 'DCM_RS_00060.dcm'),
        structure_name='GTV-1',
        reference=ct_phantom_image,
    )


@pytest.fixture()
def res3d_1mm_image_spline(ct_phantom_image):
    preprocessing = ImageResampler(
        resolution=(1, 1, 1),
        method='BSpline',
        intensity_rounding='nearest_integer',
    )
    res_image = preprocessing.apply(ct_phantom_image)

    return res_image


@pytest.fixture()
def res3d_1mm_mask_linear(ct_phantom_mask):
    preprocessing = MaskResampler(
        resolution=(1, 1, 1),
        method='Linear',
        partial_volume_threshold=0.5,
    )
    res_mask = preprocessing.apply(ct_phantom_mask)

    return res_mask


@pytest.fixture()
def checkerboard_phantom(ibsi_ii_digital_data_dir):
    return Image.from_nifti(ibsi_ii_digital_data_dir / 'nifti/checkerboard/image/checkerboard.nii.gz')


@pytest.fixture()
def impulse_phantom(ibsi_ii_digital_data_dir):
    return Image.from_nifti(ibsi_ii_digital_data_dir / 'nifti/impulse/image/impulse.nii.gz')


@pytest.fixture()
def sphere_phantom(ibsi_ii_digital_data_dir):
    return Image.from_nifti(ibsi_ii_digital_data_dir / 'nifti/sphere/image/sphere.nii.gz')


@pytest.fixture()
def pattern_1_phantom(ibsi_ii_digital_data_dir):
    return Image.from_nifti(ibsi_ii_digital_data_dir / 'nifti/pattern_1/image/pattern_1.nii.gz')


@pytest.mark.integration
@pytest.mark.parametrize(
    ('config', 'params_and_images'),
    [
        (
            case.config,
            [
                case.filter_params['padding_type'],
                case.filter_params['dimensionality'],
                f'{case.phantom}_phantom',
                case.response_map,
            ],
        )
        for case in _phase_i_cases('1.')
    ],
)
def test_ibsi_ii_ph_i_1(ibsi_ii_response_maps_dir, request, config, params_and_images):
    params_and_images = list(params_and_images)
    params_and_images[-2] = request.getfixturevalue(params_and_images[-2])
    filtering = create_filter(
        filtering_method='Mean', padding_type=params_and_images[0], dimensionality=params_and_images[1], support=15
    )
    _run_ph_i_case(filtering, params_and_images[-2], params_and_images[-1], config, ibsi_ii_response_maps_dir)


@pytest.mark.integration
@pytest.mark.parametrize(
    ('config', 'params_and_images'),
    [
        (
            case.config,
            [
                case.filter_params['padding_type'],
                case.filter_params['dimensionality'],
                case.filter_params['sigma_mm'],
                f'{case.phantom}_phantom',
                case.response_map,
            ],
        )
        for case in _phase_i_cases('2.')
    ],
)
def test_ibsi_ii_ph_i_2(ibsi_ii_response_maps_dir, request, config, params_and_images):
    params_and_images = list(params_and_images)
    params_and_images[-2] = request.getfixturevalue(params_and_images[-2])
    filtering = create_filter(
        filtering_method='Laplacian of Gaussian',
        padding_type=params_and_images[0],
        dimensionality=params_and_images[1],
        sigma_mm=params_and_images[2],
        cutoff=4,
    )
    _run_ph_i_case(filtering, params_and_images[-2], params_and_images[-1], config, ibsi_ii_response_maps_dir)


@pytest.mark.integration
@pytest.mark.parametrize(
    ('config', 'params_and_images'),
    [
        (
            case.config,
            [
                case.filter_params['padding_type'],
                case.filter_params['dimensionality'],
                case.filter_params['response_map'],
                case.filter_params['rotation_invariance'],
                case.filter_params['pooling'],
                case.filter_params['energy_map'],
                case.filter_params['distance'],
                f'{case.phantom}_phantom',
                case.response_map,
            ],
        )
        for case in _phase_i_cases('3.')
    ],
)
def test_ibsi_ii_ph_i_3(ibsi_ii_response_maps_dir, request, config, params_and_images):
    params_and_images = list(params_and_images)
    params_and_images[-2] = request.getfixturevalue(params_and_images[-2])
    filtering = create_filter(
        filtering_method='Laws Kernels',
        padding_type=params_and_images[0],
        dimensionality=params_and_images[1],
        response_map=params_and_images[2],
        rotation_invariance=params_and_images[3],
        pooling=params_and_images[4],
        energy_map=params_and_images[5],
        distance=params_and_images[6],
    )
    _run_ph_i_case(filtering, params_and_images[-2], params_and_images[-1], config, ibsi_ii_response_maps_dir)


@pytest.mark.integration
@pytest.mark.parametrize(
    "config,padding,res_mm,sigma_mm,lambda_mm,gamma,theta,rot_inv,orth_planes,n_stds,phantom,truth_file",
    [
        (
            "4.a.1",
            "constant",
            2.0,
            10.0,
            4.0,
            1 / 2,
            np.pi / 3,
            False,
            False,
            11,
            "impulse_phantom",
            "4_a_1-ValidCRM.nii",
        ),
        (
            "4.a.2",
            "constant",
            2.0,
            10.0,
            4.0,
            1 / 2,
            np.pi / 4,
            True,
            True,
            11,
            "impulse_phantom",
            "4_a_2-ValidCRM.nii",
        ),
        (
            "4.b.1",
            "reflect",
            2.0,
            20.0,
            8.0,
            5 / 2,
            5 * np.pi / 4,
            False,
            False,
            None,
            "sphere_phantom",
            "4_b_1-ValidCRM.nii",
        ),
        (
            "4.b.2",
            "reflect",
            2.0,
            20.0,
            8.0,
            5 / 2,
            np.pi / 8,
            True,
            True,
            None,
            "sphere_phantom",
            "4_b_2-ValidCRM.nii",
        ),
    ],
    ids=lambda val, *_: val,  # use the config string as the test‐id
)
def test_ibsi_ii_ph_i_4(
    config,
    padding,
    res_mm,
    sigma_mm,
    lambda_mm,
    gamma,
    theta,
    rot_inv,
    orth_planes,
    n_stds,
    phantom,
    truth_file,
    impulse_phantom,
    sphere_phantom,
    ibsi_ii_response_maps_dir,
):
    case = _phase_i_case(config)
    # Retain the published values in the parametrized test ID while making the
    # shared registry authoritative for execution.
    assert phantom == f'{case.phantom}_phantom'
    assert (padding, res_mm, sigma_mm, lambda_mm, gamma, theta, rot_inv, orth_planes, n_stds, truth_file) == (
        case.filter_params['padding_type'],
        case.filter_params['res_mm'],
        case.filter_params['sigma_mm'],
        case.filter_params['lambda_mm'],
        case.filter_params['gamma'],
        case.filter_params['theta'],
        case.filter_params['rotation_invariance'],
        case.filter_params['orthogonal_planes'],
        case.filter_params['n_stds'],
        case.response_map,
    )
    phantom_data = {'impulse': impulse_phantom, 'sphere': sphere_phantom}[case.phantom]
    filtering = create_filter(filtering_method=case.filter_method, **dict(case.filter_params))
    _run_ph_i_case(filtering, phantom_data, case.response_map, case.config, ibsi_ii_response_maps_dir)


@pytest.mark.integration
@pytest.mark.parametrize(
    ('config', 'params_and_images'),
    [
        (
            case.config,
            [
                case.filter_params['padding_type'],
                case.filter_params['response_map'],
                case.filter_params['rotation_invariance'],
                f'{case.phantom}_phantom',
                case.response_map,
            ],
        )
        for case in _phase_i_cases('5.')
    ],
)
def test_ibsi_ii_ph_i_5(ibsi_ii_response_maps_dir, request, config, params_and_images):
    params_and_images = list(params_and_images)
    params_and_images[-2] = request.getfixturevalue(params_and_images[-2])
    filtering = create_filter(
        filtering_method='Wavelets',
        wavelet_type='db2',
        dimensionality='3D',
        padding_type=params_and_images[0],
        response_map=params_and_images[1],
        decomposition_level=1,
        rotation_invariance=params_and_images[2],
    )
    _run_ph_i_case(filtering, params_and_images[-2], params_and_images[-1], config, ibsi_ii_response_maps_dir)


@pytest.mark.integration
@pytest.mark.parametrize(
    ('config', 'params_and_images'),
    [
        (
            case.config,
            [
                case.filter_params['padding_type'],
                case.filter_params['response_map'],
                case.filter_params['rotation_invariance'],
                f'{case.phantom}_phantom',
                case.response_map,
            ],
        )
        for case in _phase_i_cases('6.')
    ],
)
def test_ibsi_ii_ph_i_6(ibsi_ii_response_maps_dir, request, config, params_and_images):
    params_and_images = list(params_and_images)
    params_and_images[-2] = request.getfixturevalue(params_and_images[-2])
    filtering = create_filter(
        filtering_method='Wavelets',
        wavelet_type='coif1',
        dimensionality='3D',
        padding_type=params_and_images[0],
        response_map=params_and_images[1],
        decomposition_level=1,
        rotation_invariance=params_and_images[2],
    )
    _run_ph_i_case(filtering, params_and_images[-2], params_and_images[-1], config, ibsi_ii_response_maps_dir)


@pytest.mark.integration
@pytest.mark.parametrize(
    ('config', 'params_and_images'),
    [
        (
            case.config,
            [
                case.filter_params['padding_type'],
                case.filter_params['response_map'],
                case.filter_params['rotation_invariance'],
                f'{case.phantom}_phantom',
                case.response_map,
            ],
        )
        for case in _phase_i_cases('7.')
    ],
)
def test_ibsi_ii_ph_i_7(ibsi_ii_response_maps_dir, request, config, params_and_images):
    params_and_images = list(params_and_images)
    params_and_images[-2] = request.getfixturevalue(params_and_images[-2])
    filtering = create_filter(
        filtering_method='Wavelets',
        wavelet_type='haar',
        dimensionality='3D',
        padding_type=params_and_images[0],
        response_map=params_and_images[1],
        decomposition_level=2,
        rotation_invariance=params_and_images[2],
    )
    _run_ph_i_case(filtering, params_and_images[-2], params_and_images[-1], config, ibsi_ii_response_maps_dir)


@pytest.mark.integration
@pytest.mark.parametrize('case', _phase_i_cases('8.'), ids=lambda case: case.config)
def test_ibsi_ii_ph_i_8(ibsi_ii_response_maps_dir, checkerboard_phantom, case):
    filtering = create_filter(filtering_method=case.filter_method, **dict(case.filter_params))
    _run_ph_i_case(filtering, checkerboard_phantom, case.response_map, case.config, ibsi_ii_response_maps_dir)


@pytest.mark.integration
@pytest.mark.parametrize('case', _phase_i_cases('9.'), ids=lambda case: case.config)
def test_ibsi_ii_ph_i_9(ibsi_ii_response_maps_dir, request, case):
    filtering = create_filter(filtering_method=case.filter_method, **dict(case.filter_params))
    phantom = request.getfixturevalue(f'{case.phantom}_phantom')
    _run_ph_i_case(filtering, phantom, case.response_map, case.config, ibsi_ii_response_maps_dir)


@pytest.mark.integration
def test_ibsi_ii_ph_i_10(ibsi_ii_response_maps_dir, pattern_1_phantom):
    case = _phase_i_cases('10.')[0]
    filtering = create_filter(filtering_method=case.filter_method, **dict(case.filter_params))
    _run_ph_i_case(
        filtering,
        pattern_1_phantom,
        case.response_map,
        case.config,
        ibsi_ii_response_maps_dir,
    )


@pytest.mark.integration
def test_ibsi_ii_ph_ii_2a(ct_phantom_image, ct_phantom_mask):
    ibsi_features = ibsi_ii_feature_tolerances('2.A')

    filtering = _phase_ii_filter('2.A')

    filtered_image = filtering.apply(ct_phantom_image)

    features = _extract_filtered_features(ct_phantom_image, filtered_image, ct_phantom_mask)
    ibsi_ii_ph_ii_validation(ibsi_features, features)


@pytest.mark.integration
def test_ibsi_ii_ph_ii_2b(res3d_1mm_image_spline, res3d_1mm_mask_linear):
    ibsi_features = ibsi_ii_feature_tolerances('2.B')

    filtering = _phase_ii_filter('2.B')

    filtered_image = filtering.apply(res3d_1mm_image_spline)

    features = _extract_filtered_features(res3d_1mm_image_spline, filtered_image, res3d_1mm_mask_linear)
    ibsi_ii_ph_ii_validation(ibsi_features, features)


@pytest.mark.integration
def test_ibsi_ii_ph_ii_3a(ct_phantom_image, ct_phantom_mask):
    ibsi_features = ibsi_ii_feature_tolerances('3.A')

    filtering = _phase_ii_filter('3.A')

    filtered_image = filtering.apply(ct_phantom_image)

    features = _extract_filtered_features(ct_phantom_image, filtered_image, ct_phantom_mask)
    ibsi_ii_ph_ii_validation(ibsi_features, features)


@pytest.mark.integration
def test_ibsi_ii_ph_ii_3b(res3d_1mm_image_spline, res3d_1mm_mask_linear):
    ibsi_features = ibsi_ii_feature_tolerances('3.B')

    filtering = _phase_ii_filter('3.B')

    filtered_image = filtering.apply(res3d_1mm_image_spline)

    features = _extract_filtered_features(res3d_1mm_image_spline, filtered_image, res3d_1mm_mask_linear)
    ibsi_ii_ph_ii_validation(ibsi_features, features)


@pytest.mark.integration
def test_ibsi_ii_ph_ii_4a(ct_phantom_image, ct_phantom_mask):
    ibsi_features = ibsi_ii_feature_tolerances('4.A')

    filtering = _phase_ii_filter('4.A')

    filtered_image = filtering.apply(ct_phantom_image)

    features = _extract_filtered_features(ct_phantom_image, filtered_image, ct_phantom_mask)
    ibsi_ii_ph_ii_validation(ibsi_features, features)


@pytest.mark.integration
def test_ibsi_ii_ph_ii_4b(res3d_1mm_image_spline, res3d_1mm_mask_linear):
    ibsi_features = ibsi_ii_feature_tolerances('4.B')

    filtering = _phase_ii_filter('4.B')

    filtered_image = filtering.apply(res3d_1mm_image_spline)

    features = _extract_filtered_features(res3d_1mm_image_spline, filtered_image, res3d_1mm_mask_linear)
    ibsi_ii_ph_ii_validation(ibsi_features, features)


# Gabor
@pytest.mark.integration
def test_ibsi_ii_ph_ii_5a(ct_phantom_image, ct_phantom_mask):
    ibsi_features = ibsi_ii_feature_tolerances('5.A')

    filtering = _phase_ii_filter('5.A')

    filtered_image = filtering.apply(ct_phantom_image)

    features = _extract_filtered_features(ct_phantom_image, filtered_image, ct_phantom_mask)
    ibsi_ii_ph_ii_validation(ibsi_features, features)


@pytest.mark.integration
def test_ibsi_ii_ph_ii_5b(res3d_1mm_image_spline, res3d_1mm_mask_linear):
    ibsi_features = ibsi_ii_feature_tolerances('5.B')

    filtering = _phase_ii_filter('5.B')

    filtered_image = filtering.apply(res3d_1mm_image_spline)

    features = _extract_filtered_features(res3d_1mm_image_spline, filtered_image, res3d_1mm_mask_linear)
    ibsi_ii_ph_ii_validation(ibsi_features, features)


@pytest.mark.integration
def test_ibsi_ii_ph_ii_6a(ct_phantom_image, ct_phantom_mask):
    ibsi_features = ibsi_ii_feature_tolerances('6.A')

    filtering = _phase_ii_filter('6.A')

    filtered_image = filtering.apply(ct_phantom_image)

    features = _extract_filtered_features(ct_phantom_image, filtered_image, ct_phantom_mask)
    ibsi_ii_ph_ii_validation(ibsi_features, features)


@pytest.mark.integration
def test_ibsi_ii_ph_ii_6b(res3d_1mm_image_spline, res3d_1mm_mask_linear):
    ibsi_features = ibsi_ii_feature_tolerances('6.B')

    filtering = _phase_ii_filter('6.B')

    filtered_image = filtering.apply(res3d_1mm_image_spline)

    features = _extract_filtered_features(res3d_1mm_image_spline, filtered_image, res3d_1mm_mask_linear)
    ibsi_ii_ph_ii_validation(ibsi_features, features)


@pytest.mark.integration
def test_ibsi_ii_ph_ii_7a(ct_phantom_image, ct_phantom_mask):
    ibsi_features = ibsi_ii_feature_tolerances('7.A')

    filtering = _phase_ii_filter('7.A')

    filtered_image = filtering.apply(ct_phantom_image)

    features = _extract_filtered_features(ct_phantom_image, filtered_image, ct_phantom_mask)
    ibsi_ii_ph_ii_validation(ibsi_features, features)


@pytest.mark.integration
def test_ibsi_ii_ph_ii_7b(res3d_1mm_image_spline, res3d_1mm_mask_linear):
    ibsi_features = ibsi_ii_feature_tolerances('7.B')

    filtering = _phase_ii_filter('7.B')

    filtered_image = filtering.apply(res3d_1mm_image_spline)

    features = _extract_filtered_features(res3d_1mm_image_spline, filtered_image, res3d_1mm_mask_linear)
    ibsi_ii_ph_ii_validation(ibsi_features, features)


@pytest.mark.integration
def test_ibsi_ii_ph_ii_8a(ct_phantom_image, ct_phantom_mask):
    ibsi_features = ibsi_ii_feature_tolerances('8.A')

    filtering = _phase_ii_filter('8.A')

    filtered_image = filtering.apply(ct_phantom_image)

    features = _extract_filtered_features(ct_phantom_image, filtered_image, ct_phantom_mask)
    ibsi_ii_ph_ii_validation(ibsi_features, features)


@pytest.mark.integration
def test_ibsi_ii_ph_ii_8b(res3d_1mm_image_spline, res3d_1mm_mask_linear):
    ibsi_features = ibsi_ii_feature_tolerances('8.B')

    filtering = _phase_ii_filter('8.B')

    filtered_image = filtering.apply(res3d_1mm_image_spline)

    features = _extract_filtered_features(res3d_1mm_image_spline, filtered_image, res3d_1mm_mask_linear)
    ibsi_ii_ph_ii_validation(ibsi_features, features, config_8b=True)


@pytest.mark.integration
def test_ibsi_ii_ph_ii_9a(ct_phantom_image, ct_phantom_mask):
    ibsi_features = ibsi_ii_feature_tolerances('9.A')

    filtering = _phase_ii_filter('9.A')

    filtered_image = filtering.apply(ct_phantom_image)

    features = _extract_filtered_features(ct_phantom_image, filtered_image, ct_phantom_mask)
    ibsi_ii_ph_ii_validation(ibsi_features, features)


@pytest.mark.integration
def test_ibsi_ii_ph_ii_9b(res3d_1mm_image_spline, res3d_1mm_mask_linear):
    ibsi_features = ibsi_ii_feature_tolerances('9.B')

    filtering = _phase_ii_filter('9.B')

    filtered_image = filtering.apply(res3d_1mm_image_spline)

    features = _extract_filtered_features(res3d_1mm_image_spline, filtered_image, res3d_1mm_mask_linear)
    ibsi_ii_ph_ii_validation(ibsi_features, features)


@pytest.mark.integration
@pytest.mark.parametrize(
    ('config', 'image_fixture', 'mask_fixture'),
    [
        (
            case.config,
            'res3d_1mm_image_spline' if case.resampling_dim else 'ct_phantom_image',
            'res3d_1mm_mask_linear' if case.resampling_dim else 'ct_phantom_mask',
        )
        for case in IBSI_II_FEATURE_CASES
        if case.config.startswith('1.')
    ],
)
def test_ibsi_ii_ph_ii_unfiltered(request, config, image_fixture, mask_fixture):
    # IBSI II Table 6.3: configurations 1.A/1.B use no filter.
    image = request.getfixturevalue(image_fixture)
    mask = request.getfixturevalue(mask_fixture)
    features = _extract_filtered_features(image, image, mask)
    ibsi_ii_ph_ii_validation(ibsi_ii_feature_tolerances(config), features)
