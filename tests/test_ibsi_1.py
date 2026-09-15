from pathlib import Path

import pytest
from ibsi_cases import IBSI_I_FEATURE_CASES
from ibsi_helpers import load_references, matches_reference, select_ibsi_i_references

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


def ibsi_i_feature_tolerances(sheet_name):
    csv_path = Path(__file__).parent / 'data' / 'ibsi_1_reference_data' / f'ibsi_1_reference_values_{sheet_name}.csv'
    return load_references(csv_path, 'tag', 'reference value', delimiter=',', phase='I')


def ibsi_i_validation(ibsi_features, features):
    assert ibsi_features, "Empty IBSI reference selection"
    for raw_tag, feature_info in ibsi_features.items():
        tag = str(raw_tag)
        if tag not in features:
            pytest.fail(f"Missing required feature {tag}")

        if not matches_reference(features[tag], feature_info['reference value'], feature_info['tolerance']):
            pytest.fail(
                f"Feature {tag} out of tolerance: computed={features[tag]}, "
                f"reference={feature_info['reference value']}, tolerance={feature_info['tolerance']}"
            )


@pytest.mark.unit
@pytest.mark.parametrize('features', [{}, {'stat_mean': 1.0}])
def test_ibsi_i_requires_all_reference_features(features):
    reference = {
        'stat_mean': {'reference value': '1', 'tolerance': '0'},
        'stat_var': {'reference value': '2', 'tolerance': '0'},
    }
    missing = 'stat_var' if features else 'stat_mean'
    with pytest.raises(pytest.fail.Exception, match=f'Missing required feature {missing}'):
        ibsi_i_validation(reference, features)


@pytest.mark.unit
@pytest.mark.parametrize('value', [1 / 22, 0.0455])
def test_config_a_qcod_matches_published_precision(value):
    reference = {'ih_qcod': ibsi_i_feature_tolerances('config_A')['ih_qcod']}
    ibsi_i_validation(reference, {'ih_qcod': value})


@pytest.mark.unit
@pytest.mark.parametrize('value', [0.0454, 0.0456, float('nan'), float('inf')])
def test_config_a_qcod_rejects_mismatch(value):
    reference = {'ih_qcod': ibsi_i_feature_tolerances('config_A')['ih_qcod']}
    with pytest.raises(pytest.fail.Exception, match='out of tolerance'):
        ibsi_i_validation(reference, {'ih_qcod': value})


@pytest.mark.unit
def test_config_a_qcod_requires_feature():
    reference = {'ih_qcod': ibsi_i_feature_tolerances('config_A')['ih_qcod']}
    with pytest.raises(pytest.fail.Exception, match='Missing required feature ih_qcod'):
        ibsi_i_validation(reference, {})


@pytest.fixture()
def dcm_ct_phantom_image(ibsi_ct_data_dir):
    return Image.from_dicom(dicom_dir=ibsi_ct_data_dir / 'dicom' / 'image', modality='CT')


@pytest.fixture()
def dcm_ct_phantom_mask(dcm_ct_phantom_image, ibsi_ct_data_dir):
    return Image.from_dicom_mask(
        reference=dcm_ct_phantom_image,
        rtstruct_path=str(ibsi_ct_data_dir / 'dicom' / 'mask' / 'DCM_RS_00060.dcm'),
        structure_name='GTV-1',
    )


@pytest.fixture()
def nii_ct_phantom_image(ibsi_ct_data_dir):
    return Image.from_nifti(str(ibsi_ct_data_dir / 'nifti' / 'image' / 'phantom.nii.gz'))


@pytest.fixture()
def nii_ct_phantom_mask(nii_ct_phantom_image, ibsi_ct_data_dir):
    return Image.from_nifti_mask(
        reference=nii_ct_phantom_image,
        mask_path=str(ibsi_ct_data_dir / 'nifti' / 'mask' / 'mask.nii.gz'),
    )


def _resolution(image, value, dimension):
    if dimension == '2D':
        return (value, value, image.spacing[2])
    return (value, value, value)


def _prepare_roi_data(
    image,
    mask,
    intensity_range=None,
    outlier_range=None,
    number_of_bins=None,
    bin_size=None,
    ivh_method=None,
    ivh_number_of_bins=None,
    ivh_bin_size=None,
):
    roi_data = IntensityMaskBuilder().apply(
        RoiData(
            image=image,
            morphological_mask=mask,
        )
    )
    roi_data = Resegmenter(
        intensity_range=intensity_range,
        outlier_range=outlier_range,
    ).apply(roi_data)
    if number_of_bins is not None or bin_size is not None:
        roi_data = TextureDiscretizer(
            number_of_bins=number_of_bins,
            bin_size=bin_size,
        ).apply(roi_data)
    if ivh_method is not None:
        roi_data = IVHIntensityDiscretizer(
            method=ivh_method,
            number_of_bins=ivh_number_of_bins,
            bin_size=ivh_bin_size,
        ).apply(roi_data)
    return roi_data


def _extract_features(image, mask, aggr_dim, aggr_method, families=None, **prep_kwargs):
    return Radiomics(
        aggr_dim=aggr_dim,
        aggr_method=aggr_method,
    ).extract_features(
        families=families,
        roi_data=_prepare_roi_data(image, mask, **prep_kwargs),
    )


@pytest.fixture()
def res2d_2mm_image_linear(nii_ct_phantom_image):

    preprocessing = ImageResampler(
        resolution=_resolution(nii_ct_phantom_image, 2, '2D'),
        method='Linear',
        intensity_rounding='nearest_integer',
    )
    res_image = preprocessing.apply(nii_ct_phantom_image)

    return res_image


@pytest.fixture()
def res2d_2mm_mask_linear(nii_ct_phantom_mask):

    preprocessing = MaskResampler(
        resolution=_resolution(nii_ct_phantom_mask, 2, '2D'),
        method='Linear',
        partial_volume_threshold=0.5,
    )
    res_mask = preprocessing.apply(nii_ct_phantom_mask)

    return res_mask


@pytest.fixture()
def res3d_2mm_image_linear(nii_ct_phantom_image):

    preprocessing = ImageResampler(
        resolution=_resolution(nii_ct_phantom_image, 2, '3D'),
        method='Linear',
        intensity_rounding='nearest_integer',
    )
    res_image = preprocessing.apply(nii_ct_phantom_image)

    return res_image


@pytest.fixture()
def res3d_2mm_mask_linear(nii_ct_phantom_mask):

    preprocessing = MaskResampler(
        resolution=_resolution(nii_ct_phantom_mask, 2, '3D'),
        method='Linear',
        partial_volume_threshold=0.5,
    )
    res_mask = preprocessing.apply(nii_ct_phantom_mask)

    return res_mask


@pytest.fixture()
def res3d_2mm_image_spline(dcm_ct_phantom_image):

    preprocessing = ImageResampler(
        resolution=_resolution(dcm_ct_phantom_image, 2, '3D'),
        method='BSpline',
        intensity_rounding='nearest_integer',
    )
    res_image = preprocessing.apply(dcm_ct_phantom_image)

    return res_image


def _ibsi_i_cases(config):
    return tuple(case for case in IBSI_I_FEATURE_CASES if case.config == config)


def _run_ibsi_i_feature_case(case, image, mask, families=None):
    reference_name = 'digital_phantom' if case.config == 'digital' else f'config_{case.config}'
    reference = select_ibsi_i_references(ibsi_i_feature_tolerances(reference_name), *case.aggregation)
    features = _extract_features(image, mask, *case.aggregation, families=families, **dict(case.preparation))
    ibsi_i_validation(reference, features)


@pytest.mark.integration
@pytest.mark.parametrize('case', _ibsi_i_cases('A'), ids=lambda case: case.identifier)
def test_ibsi_i_config_a(dcm_ct_phantom_image, dcm_ct_phantom_mask, case):
    _run_ibsi_i_feature_case(case, dcm_ct_phantom_image, dcm_ct_phantom_mask)


@pytest.mark.integration
@pytest.mark.parametrize('case', _ibsi_i_cases('B'), ids=lambda case: case.identifier)
def test_ibsi_i_config_b(res2d_2mm_image_linear, res2d_2mm_mask_linear, case):
    _run_ibsi_i_feature_case(case, res2d_2mm_image_linear, res2d_2mm_mask_linear)


@pytest.mark.integration
@pytest.mark.parametrize('case', _ibsi_i_cases('C'), ids=lambda case: case.identifier)
def test_ibsi_i_config_c(res3d_2mm_image_linear, res3d_2mm_mask_linear, case):
    _run_ibsi_i_feature_case(case, res3d_2mm_image_linear, res3d_2mm_mask_linear)


@pytest.mark.integration
@pytest.mark.parametrize('case', _ibsi_i_cases('D'), ids=lambda case: case.identifier)
def test_ibsi_i_config_d(res3d_2mm_image_linear, res3d_2mm_mask_linear, case):
    _run_ibsi_i_feature_case(case, res3d_2mm_image_linear, res3d_2mm_mask_linear)


@pytest.mark.integration
@pytest.mark.parametrize('case', _ibsi_i_cases('E'), ids=lambda case: case.identifier)
def test_ibsi_i_config_e(res3d_2mm_image_spline, res3d_2mm_mask_linear, case):
    _run_ibsi_i_feature_case(case, res3d_2mm_image_spline, res3d_2mm_mask_linear)


@pytest.mark.integration
@pytest.mark.parametrize('case', _ibsi_i_cases('digital'), ids=lambda case: case.identifier)
def test_ibsi_i_digital_phantom(ibsi_i_digital_data_dir, case):
    image = Image.from_nifti(ibsi_i_digital_data_dir / 'nifti' / 'image' / 'phantom.nii.gz')
    mask = Image.from_nifti(ibsi_i_digital_data_dir / 'nifti' / 'mask' / 'mask.nii.gz')
    _run_ibsi_i_feature_case(case, image, mask, families='all')


def _diagnostic_values(image, roi, stage):
    import numpy as np

    result = {}
    if stage != 'reseg':
        for axis, size, spacing in zip('xyz', image.array.shape[::-1], image.spacing):
            result[f'img_dim_{axis}_{stage}_img'] = size
            result[f'vox_dim_{axis}_{stage}_img'] = spacing
        for name, operation in [('mean', np.mean), ('min', np.min), ('max', np.max)]:
            result[f'{name}_int_{stage}_img'] = float(operation(image.array))
    intensity = roi.intensity_mask.array
    for kind, mask in [('int', np.isfinite(intensity)), ('morph', roi.morphological_mask.array > 0)]:
        positions = np.argwhere(mask)
        dimensions = (positions.max(axis=0) - positions.min(axis=0) + 1)[::-1]
        for axis, size in zip('xyz', dimensions):
            result[f'{kind}_mask_bb_dim_{axis}_{stage}_roi'] = size
        result[f'{kind}_mask_vox_count_{stage}_roi'] = int(mask.sum())
    for axis, size in zip('xyz', intensity.shape[::-1]):
        result[f'int_mask_dim_{axis}_{stage}_roi'] = size
    for name, operation in [('mean', np.nanmean), ('min', np.nanmin), ('max', np.nanmax)]:
        result[f'int_mask_{name}_int_{stage}_roi'] = float(operation(intensity))
    return result


@pytest.mark.integration
@pytest.mark.parametrize('config', ['A', 'B', 'C', 'D', 'E'])
@pytest.mark.parametrize('stage', ['init', 'interp', 'reseg'])
def test_ibsi_i_diagnostics(request, config, stage):
    image_name, mask_name = {
        'A': ('dcm_ct_phantom_image', 'dcm_ct_phantom_mask'),
        'B': ('res2d_2mm_image_linear', 'res2d_2mm_mask_linear'),
        'C': ('res3d_2mm_image_linear', 'res3d_2mm_mask_linear'),
        'D': ('res3d_2mm_image_linear', 'res3d_2mm_mask_linear'),
        'E': ('res3d_2mm_image_spline', 'res3d_2mm_mask_linear'),
    }[config]
    if stage == 'init':
        image_name, mask_name = 'dcm_ct_phantom_image', 'dcm_ct_phantom_mask'
    image, mask = request.getfixturevalue(image_name), request.getfixturevalue(mask_name)
    roi = IntensityMaskBuilder().apply(RoiData(image=image, morphological_mask=mask))
    if stage == 'reseg':
        ranges = {'A': [-500, 400], 'B': [-500, 400], 'C': [-1000, 400], 'D': None, 'E': [-1000, 400]}
        roi = Resegmenter(intensity_range=ranges[config], outlier_range=3 if config in ('D', 'E') else None).apply(roi)
    values = _diagnostic_values(image, roi, stage)
    reference = {
        tag: row
        for tag, row in ibsi_i_feature_tolerances(f'config_{config}').items()
        if row['family'].startswith('Diagnostics') and f'_{stage}_' in tag
    }
    ibsi_i_validation(reference, values)


@pytest.mark.integration
@pytest.mark.parametrize('config', list('ABCDE'))
@pytest.mark.parametrize('method', ['fft', 'blocked'])
def test_ibsi_i_morphology_correlation(config, method, request, monkeypatch):
    from zrad.radiomics import morphology

    fixture_names = {
        'A': ('dcm_ct_phantom_image', 'dcm_ct_phantom_mask'),
        'B': ('res2d_2mm_image_linear', 'res2d_2mm_mask_linear'),
        'C': ('res3d_2mm_image_linear', 'res3d_2mm_mask_linear'),
        'D': ('res3d_2mm_image_linear', 'res3d_2mm_mask_linear'),
        'E': ('res3d_2mm_image_spline', 'res3d_2mm_mask_linear'),
    }
    image, mask = (request.getfixturevalue(name) for name in fixture_names[config])
    roi = _prepare_roi_data(
        image,
        mask,
        intensity_range=[-500, 400] if config in 'AB' else [-1000, 400] if config in 'CE' else None,
        outlier_range=3 if config in 'DE' else None,
    )
    original_selector = morphology._correlation_method

    def select(n, shape):
        assert original_selector(n, shape) == 'fft'
        # Validate both exact algorithms on the same reference fixtures.
        return method

    monkeypatch.setattr(morphology, '_correlation_method', select)
    features = Radiomics().extract_features(roi_data=roi, families=['morphology'])
    tags = {'morph_moran_i', 'morph_geary_c'}
    assert set(features) == set(morphology.MORPHOLOGY_FEATURE_NAMES)
    references = ibsi_i_feature_tolerances('config_' + config)
    ibsi_i_validation({tag: references[tag] for tag in tags}, features)
