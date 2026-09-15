=======================
IBSI Benchmark Coverage
=======================

Z-Rad implements IBSI preprocessing, features, and filters and tests them
against published reference data. This page explains what is implemented,
which benchmark configurations are tested, and how to interpret agreement.
Implementation coverage describes available operations; benchmark validation
provides numerical evidence for specific configurations.

Implementation Coverage
-----------------------

Z-Rad supports all IBSI I preprocessing operations and radiomic features,
and all IBSI II filters. The guides below describe the available settings.

.. list-table:: Implementation guides
   :header-rows: 1
   :widths: 25 50 25

   * - Area
     - Available operations
     - Guide
   * - IBSI I preprocessing
     - Image and mask interpolation, resegmentation, and intensity discretization.
     - :doc:`../user/preprocessing`
   * - IBSI I features
     - Morphology, local intensity, intensity statistics, histograms,
       intensity-volume histograms, and texture.
     - :doc:`../user/radiomics` and :doc:`../reference/radiomics`
   * - IBSI II filters
     - Mean, LoG, Laws, Gabor, separable wavelets, Simoncelli wavelets,
       and Riesz transforms.
     - :doc:`../user/filtering`

.. _ibsi-benchmark-coverage:

Benchmark Coverage
------------------

The matrix describes the automated checks available in the repository.
Results for a particular revision are provided in the execution reports
linked under `Results and Reproduction`_.

For IBSI I, **features per mode** counts the feature comparisons within one
aggregation mode. **Distinct reference tags** counts unique reference-table
identifiers across the listed modes, including aggregation suffixes; some
features share a reference tag across modes. For IBSI II phase II,
**feature comparisons** counts feature/configuration pairs. These totals
should not be interpreted as counts of independent feature definitions.

.. list-table:: Automated benchmark coverage
   :header-rows: 1
   :widths: 23 52 25

   * - Benchmark
     - Coverage
     - Limitations
   * - IBSI I digital phantom
     - 6 aggregation modes; 169 features per mode; 482 distinct reference tags.
     - Features without references are listed below.
   * - IBSI I CT diagnostics A--E
     - All 60 diagnostic rows per configuration at initial, interpolated,
       and resegmented stages: image and ROI dimensions, voxel spacing,
       bounding boxes, voxel counts, and intensities.
     - Reported separately from feature comparisons.
   * - IBSI I CT A and B
     - 4 aggregation modes each; 169 features per mode;
       346 distinct reference tags per configuration.
     - Features without references are listed below.
   * - IBSI I CT C, D, and E
     - 2 aggregation modes each; 169 features per mode;
       210 distinct reference tags per configuration.
     - Features without references are listed below.
   * - IBSI II phase I
     - All 33 bundled published filter response maps; one case per map.
     - Response-map agreement does not establish phase II feature agreement.
   * - IBSI II phase II 1.A--9.B
     - 18 configurations; 323 feature comparisons on filtered CT images.
     - One feature in 8.B has no consensus reference; see below.
   * - IBSI-SUV
     - 43 valid and 15 intentionally invalid digital reference objects;
       ROI minimum, median, and maximum SUV for valid objects.
     - SUV checks are separate from radiomic-feature comparisons.

The performance ``exhaustive`` tier measures all 20 IBSI I
feature/aggregation cases, 33 IBSI II phase-I response maps, and 18 IBSI II
phase-II feature configurations through the shared ``tests/ibsi_cases.py``
registry. These timing and memory measurements are separate from numerical
agreement checks. See :doc:`../developer/benchmarking` for the methodology.

Reference Limitations
---------------------

The following exclusions arise from unavailable reference values or
tolerances. Passing the available comparisons does not establish agreement
for these excluded features or configurations.

.. list-table:: Unavailable references
   :header-rows: 1
   :widths: 25 50 25

   * - Benchmark
     - Excluded features or configurations
     - Reference availability
   * - IBSI I
     - Volume and area density of the oriented minimum bounding box
       (``morph_vol_dens_ombb``, ``morph_area_dens_ombb``); volume and area
       density of the minimum volume enclosing ellipsoid
       (``morph_vol_dens_mvee``, ``morph_area_dens_mvee``); area under the
       intensity-volume histogram curve (``ivh_auc``).
     - The five rows have blank values and tolerances in the bundled tables.
   * - IBSI II phase II 8.B
     - Quartile coefficient of dispersion (``stat_qcod``) for 3D Simoncelli
       filtering, decomposition level 1. The other 17 features are compared.
     - IBSI reports no consensus for this feature/configuration pair.
   * - IBSI II phase II 10.A/B and 11.A/B
     - All 18 features in each configuration.
     - These configurations are absent from the published feature-reference CSV.

The 8.B exception is documented in Table 7.16 of the `IBSI II reference manual
<https://doi.org/10.48550/arXiv.2006.05470>`_. The bundled CSV matches the
`official IBSI II reference data
<https://github.com/theibsi/ibsi_2_reference_data/blob/main/reference_feature_values/reference_values.csv>`_,
which cover phase II configurations 1.A--9.B. Phase I response maps are
separate references; configuration numbers from the two phases are not
interchangeable.

The suite does not cover the IBSI II phase III clinical validation study or
establish universal compliance across all inputs and processing options.

.. _ibsi-comparison-rules:

Comparison Rules
----------------

All scalar reference-table comparisons use the same policy, including IBSI I
features and preprocessing diagnostics and IBSI II phase II features:

* **Positive tolerance:** the unrounded result must lie in the inclusive
  interval ``[reference - tolerance, reference + tolerance]``.
* **Zero tolerance:** the result must agree exactly after rounding to the
  reference precision, using at least three significant figures and retaining
  any finer precision in the reference text. For example, ``0.0454545...``
  matches ``0.0455``. A zero reference with zero tolerance requires exact zero.
* **Required results:** missing expected features, NaN, and infinite results
  fail validation, subject to the documented reference exclusions above.

These rules affect benchmark comparisons only; extracted values and reference
files are unchanged. See :ref:`ibsi-reference-validation` for reference-loading
checks and precision edge cases.

IBSI II phase I requires matching response-map shapes, finite values, and
voxel-wise agreement within 1% of the reference map's intensity range.
IBSI-SUV compares valid cases to two decimal places and requires exceptions
for intentionally invalid cases. Neither uses scalar reference-table tolerances.

Supplemental Checks
-------------------

An additional 53 cases check geometry, DICOM loading, and filter behavior
using IBSI II digital assets. These checks provide software regression
coverage without adding published consensus comparisons. Reports list them
separately from benchmarks. See :ref:`ibsi-supplemental-checks` for the case
breakdown and test methodology.

Results and Reproduction
------------------------

CI publishes per-case IBSI execution reports and JUnit results in
``ibsi-results-python-*`` artifacts for each Python version. Inspect the
passed, failed, and skipped cases for the revision of interest; the coverage
matrix above describes scope, not a current test result. Line coverage is a
separate metric.

Follow :ref:`ibsi-benchmark-reports` for commands and report metadata guidance.
For reproducible studies, retain the Z-Rad version, image and mask geometry,
and exact preprocessing, filtering, discretization, and aggregation settings
alongside the extracted feature table.

Reference Tests, Data, and Licensing
------------------------------------

* `IBSI I tests <https://github.com/medical-physics-usz/z-rad/blob/master/tests/test_ibsi_1.py>`_
  define the digital-phantom and CT A--E configurations and aggregation modes.
  The `IBSI I reference tables <https://github.com/medical-physics-usz/z-rad/tree/master/tests/data/ibsi_1_reference_data>`_
  provide the expected values and tolerances.
* `IBSI II tests <https://github.com/medical-physics-usz/z-rad/blob/master/tests/test_ibsi_2.py>`_
  define the phase I filter settings and response-map comparisons, and
  phase II CT feature comparisons. The `IBSI II reference data <https://github.com/medical-physics-usz/z-rad/tree/master/tests/data/ibsi_2_reference_data>`_
  contain the response-map archive and phase II feature table.
* `IBSI-SUV tests <https://github.com/medical-physics-usz/z-rad/blob/master/tests/test_pet_suv.py>`_
  define the valid and intentionally invalid digital reference cases.

The bundled datasets use multiple open licenses depending on the component.
See `tests/data/README.md
<https://github.com/medical-physics-usz/z-rad/blob/master/tests/data/README.md>`_
for dataset locations, attribution, and license terms.
