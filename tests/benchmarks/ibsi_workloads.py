"""Registry-driven exhaustive IBSI timing and memory workloads."""

from pathlib import Path

import numpy as np
from ibsi_cases import IbsiFeatureCase, IbsiFilterCase
from ibsi_helpers import load_references, matches_reference, select_ibsi_i_references

from zrad.filtering import create_filter
from zrad.image import Image
from zrad.preprocessing import (
    ImageResampler,
    IntensityMaskBuilder,
    IVHIntensityDiscretizer,
    MaskResampler,
    Pipeline,
    Resegmenter,
    RoiData,
    TextureDiscretizer,
)
from zrad.radiomics import Radiomics

from .workloads import Workload, clear_radiomics_result_caches, metadata

DATA = Path(__file__).resolve().parents[1] / 'data'


def _read_only(*images):
    for image in images:
        image.array.flags.writeable = False
    return images


def load_ct_sources(directory):
    dcm_image = Image.from_dicom(directory / 'dicom/image', modality='CT')
    dcm_mask = Image.from_dicom_mask(
        reference=dcm_image,
        rtstruct_path=directory / 'dicom/mask/DCM_RS_00060.dcm',
        structure_name='GTV-1',
    )
    nii_image = Image.from_nifti(directory / 'nifti/image/phantom.nii.gz')
    nii_mask = Image.from_nifti_mask(reference=nii_image, mask_path=directory / 'nifti/mask/mask.nii.gz')
    _read_only(dcm_image, dcm_mask, nii_image, nii_mask)
    return {
        'ct_dicom': (dcm_image, dcm_mask),
        'ct_nifti': (nii_image, nii_mask),
    }


def load_ibsi_i_digital(directory):
    image = Image.from_nifti(directory / 'nifti/image/phantom.nii.gz')
    mask = Image.from_nifti(directory / 'nifti/mask/mask.nii.gz')
    return {'i_digital': _read_only(image, mask)}


def load_ibsi_ii_phantoms(directory):
    phantoms = {}
    for name in ('checkerboard', 'impulse', 'sphere', 'pattern_1'):
        phantoms[name] = Image.from_nifti(directory / f'nifti/{name}/image/{name}.nii.gz')
        phantoms[name].array.flags.writeable = False
    return phantoms


def _resolution(image, value, dimension):
    return (value, value, image.spacing[2] if dimension == '2D' else value)


def _pipeline(case, image):
    steps = []
    if case.resampling_dim:
        value = 2 if case.phase == 'I' else 1
        resolution = _resolution(image, value, case.resampling_dim)
        steps.extend(
            [
                (
                    'image',
                    ImageResampler(
                        resolution=resolution,
                        method=case.image_interpolation,
                        intensity_rounding='nearest_integer',
                    ),
                ),
                ('mask', MaskResampler(resolution=resolution, method='Linear', partial_volume_threshold=0.5)),
            ]
        )
    if case.filter_method:
        steps.append(('filter', create_filter(filtering_method=case.filter_method, **dict(case.filter_params))))
    steps.extend(
        [
            ('roi', IntensityMaskBuilder()),
            (
                'range',
                Resegmenter(
                    intensity_range=case.preparation.get('intensity_range'),
                    outlier_range=case.preparation.get('outlier_range'),
                ),
            ),
        ]
    )
    if case.preparation.get('number_of_bins') is not None or case.preparation.get('bin_size') is not None:
        steps.append(
            (
                'texture',
                TextureDiscretizer(
                    number_of_bins=case.preparation.get('number_of_bins'),
                    bin_size=case.preparation.get('bin_size'),
                ),
            )
        )
    if case.preparation.get('ivh_method') is not None:
        steps.append(
            (
                'ivh',
                IVHIntensityDiscretizer(
                    method=case.preparation['ivh_method'],
                    number_of_bins=case.preparation.get('ivh_number_of_bins'),
                    bin_size=case.preparation.get('ivh_bin_size'),
                ),
            )
        )
    return Pipeline(steps)


def _feature_references(case):
    if case.phase == 'I':
        name = 'digital_phantom' if case.config == 'digital' else f'config_{case.config}'
        references = load_references(
            DATA / f'ibsi_1_reference_data/ibsi_1_reference_values_{name}.csv',
            'tag',
            'reference value',
            delimiter=',',
            phase='I',
        )
        return select_ibsi_i_references(references, *case.aggregation), 'reference value'
    references = load_references(
        DATA / 'ibsi_2_reference_data/reference_feature_values/reference_values.csv',
        'feature_tag',
        'consensus_value',
        delimiter=';',
        phase='II',
        config=case.config,
    )
    return references, 'consensus_value'


def ibsi_feature(case: IbsiFeatureCase, sources):
    image = sources[case.image_source][0]
    mask = sources[case.mask_source][1]
    pipeline = _pipeline(case, image)
    extractor = Radiomics(aggr_dim=case.aggregation[0], aggr_method=case.aggregation[1])
    references, value_key = _feature_references(case)
    roi = RoiData(image=image, morphological_mask=mask)

    def operation():
        prepared = pipeline.apply(roi)
        return extractor.extract_features(roi_data=prepared, families='all' if case.config == 'digital' else None)

    def validate(result):
        assert references
        for tag, row in references.items():
            if case.phase == 'II' and case.config == '8.B' and tag == 'stat_qcod':
                assert row[value_key] == row['tolerance'] == ''
                continue
            assert tag in result, tag
            assert matches_reference(result[tag], row[value_key], row['tolerance']), (tag, result[tag], row)

    return Workload(
        case.identifier,
        operation,
        validate,
        metadata(
            image,
            mask,
            tier='exhaustive',
            ibsi_phase=case.phase,
            ibsi_config=case.config,
            steps=pipeline.get_params(),
            aggregation='/'.join(case.aggregation),
            reference_features=len(references),
            seed=None,
        ),
        rounds=3,
        setup=clear_radiomics_result_caches,
    )


def ibsi_filter(case: IbsiFilterCase, phantoms, response_maps: Path):
    image = phantoms[case.phantom]
    expected = Image.from_nifti(response_maps / 'reference_response_maps' / case.response_map).array
    filtering = create_filter(filtering_method=case.filter_method, **dict(case.filter_params))

    def validate(result):
        actual = result.array
        assert actual.shape == expected.shape and actual.size
        assert np.isfinite(actual).all() and np.isfinite(expected).all()
        tolerance = 0.01 * np.ptp(expected)
        assert np.count_nonzero(np.abs(actual - expected) > tolerance) == 0, case.config

    return Workload(
        case.identifier,
        lambda: filtering.apply(image),
        validate,
        metadata(
            image,
            tier='exhaustive',
            ibsi_phase='II phase I',
            ibsi_config=case.config,
            filter=filtering.get_params(),
            response_map=case.response_map,
            seed=None,
        ),
        rounds=3,
    )
