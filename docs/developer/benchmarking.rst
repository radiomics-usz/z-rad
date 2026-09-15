Performance benchmarking
========================

The suite in ``tests/benchmarks`` measures Z-Rad computation, independently of
correctness-test duration. Use operation benchmarks to locate a change and the
IBSI workflows to assess its effect on realistic processing. Production code
is unchanged; correctness expectations and published tolerances are preserved.

Coverage tiers and focused suites
----------------------------------

The named suites balance feedback time, diagnosis and configuration coverage.
The focused ``ibsi`` suite is available in addition to the two full tiers:

``standard``
    Scaling inputs, expensive filter/aggregation paths, and all 20 IBSI I
    workflows. This is the local launcher default.
``exhaustive``
    Every standard case plus all 33 IBSI II phase-I response maps and all 18
    IBSI II phase-II feature configurations, completing the 71 published IBSI
    cases exercised by the correctness suite.
``ibsi``
    The 71 published IBSI performance cases only, grouped as ``ibsi1`` (20
    IBSI I cases), ``ibsi2_phase1`` (33 IBSI II phase-I cases), and
    ``ibsi2_phase2`` (18 IBSI II phase-II cases).

The declarations in ``tests/ibsi_cases.py`` are shared with IBSI correctness
tests and exhaustive performance construction. Registry contract tests require
the published 20/33/18 inventory and 71 unique workload identifiers.

Install and run
---------------

Install the normal test extra (also included in ``dev``)::

    python -m pip install -e '.[test]'

``pytest-benchmark>=5.3.0`` is intentional: 5.3.0 was the current stable release
checked on 2026-09-11 and includes pytest 9 compatibility and current xdist
detection fixes. Candidate percentage changes are calculated explicitly by this
framework, independently of pytest-benchmark's file ordering. There is no
exact pin. ``threadpoolctl>=3.5`` controls supported BLAS/OpenMP backends, including
SciPy's bundled OpenBLAS. Neither dependency is required by ordinary Z-Rad users.
See the `pytest-benchmark changelog
<https://pytest-benchmark.readthedocs.io/en/latest/changelog.html>`_.

Normal tests retain their coverage and xdist defaults::

    python -m pytest
    python -m pytest -m unit
    python -m pytest -m integration

``pytest.ini`` sets ``--benchmark-skip``. The plugin skips benchmark tests before
fixtures run, so ordinary tests neither generate benchmark images nor extract
benchmark datasets. Benchmarks do not have ``unit`` or ``integration`` markers.

For timing, the recommended portable launcher sets native thread environment
variables **before** importing numerical libraries, then invokes native pytest::

    # Standard benchmark suite (the local default)
    python tests/benchmarks/run.py --suite standard

    # Standard cases plus the IBSI II phase-I and phase-II workflows
    python tests/benchmarks/run.py --suite exhaustive

    # Published IBSI configuration matrix only
    python tests/benchmarks/run.py --suite ibsi

The equivalent direct pytest commands are::

    python -m pytest tests/benchmarks --benchmark-only -n 0 --no-cov -m 'not benchmark_exhaustive'
    python -m pytest tests/benchmarks --benchmark-only -n 0 --no-cov -m 'benchmark_ibsi'
    python -m pytest tests/benchmarks --benchmark-only -n 0 --no-cov

``--benchmark-only`` overrides the normal native skip. ``-n 0`` overrides
``-n auto``; competing pytest workers would otherwise measure CPU contention.
``--no-cov`` overrides the coverage configuration; instrumentation changes the
work being measured. The suite rejects intentional timing with workers, coverage,
explicit cProfile/pytest-memray profiling, disabled garbage collection, or active
Python tracing/profiling. Do not wrap these commands in external profilers.

Groups and exact measured regions
---------------------------------

Each factory in ``workloads.py`` prepares inputs and returns a callable plus a
validator and, where needed, per-round setup. ``measure`` invokes only the
operation callable inside ``benchmark.pedantic``'s timed region. Per-round setup
runs before each warmup and measured call, outside timing. Validation happens
after measurement. The shared definitions also power memory measurement without
importing the timing fixture.

.. list-table:: Workloads
   :header-rows: 1
   :widths: 18 44 38

   * - Group
     - Measured operation
     - Outside timing
   * - image
     - ``Image.resample_to_target`` with linear interpolation at three sizes on
       both a source-derived grid and an independently constructed grid with a
       shifted origin, different field of view, and partial source overlap;
       includes the API's array/SimpleITK conversions and output construction
     - Source and target grid construction; target geometry and background-fill
       validation
   * - preprocessing
     - Isotropic ``ImageResampler.apply`` and ``MaskResampler.apply`` with
       nearest-neighbor, linear, B-spline, and Gaussian interpolation at medium
       size; linear and B-spline image scaling at small and large sizes;
       medium in-plane linear image and nearest-neighbor/linear mask cases;
       separately intensity ROI construction, range/outlier re-segmentation,
       32-bin fixed-bin-number texture discretization, and fixed-bin-size
       texture discretization with width 25 and range anchor -50
     - Synthetic image/mask generation; resampler construction; preceding ROI
       preparation for the individual re-segmentation/discretization cases.
       Fixed-bin-size texture preparation includes range re-segmentation before
       timing, so only ``TextureDiscretizer.apply`` is measured.
       In-plane cases preserve the source z spacing on a multi-slice input;
       both modes use the same 3D SimpleITK filter with different output grids.
       Output shape and voxel count are recorded for each case
   * - filtering
     - ``apply`` for matched 2D/3D Mean and LoG paths, rotation and level
       choices for db3 wavelets, plain versus rotation-invariant energy-map
       Laws, fixed versus rotated/three-plane Gabor on the same small input,
       periodic/nearest and 2D/3D Simoncelli, and first-/second-order Riesz-LoG
       with optional structure-tensor alignment
     - Synthetic image generation and filter construction. Gabor kernel caches
       are cleared before every warmup and measured call, outside timing;
       kernel generation during ``apply`` is included. Array conversions and
       output construction performed by ``apply`` remain included
   * - radiomics
     - Complete fresh-image ``families='all'`` extraction and selected Moran's
       I/Geary's C at medium size; all 11 feature families separately; every IBSI
       texture aggregation path
     - Intensity-mask building, re-segmentation, texture and IVH discretization,
       extractor construction; local-means result-cache reset before each complete
       extraction round. Extractor-internal mask validation/copying remains included
   * - ibsi1
     - All 20 published IBSI I feature/aggregation workflows, including their
       configured preprocessing and radiomics extraction
     - Digital-phantom or CT/RTSTRUCT loading, pipeline/extractor construction,
       reference CSV loading, and published-tolerance checks
   * - ibsi2_phase1
     - ``apply`` for all 33 published IBSI II phase-I filter response maps
     - Digital-phantom and reference response-map loading, filter construction,
       and response-map validation
   * - ibsi2_phase2
     - All 18 published IBSI II phase-II feature workflows, including configured
       resampling, filtering, ROI preparation, and radiomics extraction
     - CT/RTSTRUCT loading, pipeline/extractor construction, reference CSV
       loading, and published-tolerance checks

``standard`` includes ``ibsi1``; ``exhaustive`` adds ``ibsi2_phase1`` and
``ibsi2_phase2``. The focused ``ibsi`` suite runs all three groups. They use the
repository's digital phantoms and CT phantom, with
GTV-1 RTSTRUCT for CT workflows. The 71 cases come from the shared registry;
input, CSV, and response-map loading is outside timing, and published-reference
validation happens after measurement. Configuration C and 3.B are selectable
through their published registry workload IDs in both timing and memory.

Sizes and scaling
-----------------

Array shapes are **(z, y, x)**; geometry spacing is **(x, y, z)**.

.. list-table:: Deterministic synthetic inputs
   :header-rows: 1

   * - Size
     - Resampling/filtering volume
     - Radiomics volume
   * - small
     - 32 x 96 x 96 (294,912 voxels)
     - 16 x 24 x 24 (9,216 voxels)
   * - medium
     - 64 x 128 x 128 (1,048,576 voxels)
     - 24 x 40 x 40 (38,400 voxels)
   * - large (slow)
     - 96 x 192 x 192 (3,538,944 voxels)
     - 40 x 64 x 64 (163,840 voxels)

All use float64 data, spacing (1, 1, 2) mm, a fixed RNG seed (20260911), smooth
structure plus nonconstant noise, and an ellipsoidal ROI covering approximately
23% of the volume. Actual voxel count/fraction is stored. The image sizes exercise
megabyte-scale allocation and 3D computation; ROI-specific volumes bound expensive
texture work while scaling from about 2,000 to 38,000 ROI voxels. These are selective
patch/ROI workloads, not a claim to represent every scanner's full field of view.
Real CT workflows supply the complementary larger, irregular workload.
The timing and memory suites run complete fresh extraction and spatial statistics
only at medium radiomics size. Small and large synthetic radiomics inputs remain
available to workload factories for future scaling studies.

Stable ``extra_info.workload_id`` identifiers match timing and memory results.
Metadata records shapes, ROI size/fraction, spacing, dtype, parameters, seed,
aggregation, requested/observed thread controls, dependencies, loaded Z-Rad path,
and a hash of the benchmark Python files. Change the suite version and workload
identifier if the measured region, data distribution, or parameters change.
Do not compare incompatible workloads just because test names match.

Rounds, state, threads, and statistics
--------------------------------------

Most operations run seven measured calls; expensive scaling/radiomics cases
use five; IBSI I feature workflows, IBSI II phase-I filters, and IBSI II
phase-II feature workflows use three. Each has one
unmeasured warmup and one iteration per round. This bounds the costly 3D work.
Initial measurements found the fast suite took seconds, whereas a single IBSI II
call took tens of seconds; automatic calibration of that workflow would provide
little benefit.
Three-round workloads give only a preliminary distribution, not a precise
confidence bound.
To characterize noise, repeat entire runs and, if justified, increase the specific
workload's rounds. Native ``--benchmark-min-rounds`` does not override pedantic rounds.

Input arrays are read-only; preprocessing/filtering return new outputs, and the
extractor builds a fresh context and feature groups per call. Complete radiomics
extraction additionally resets ``_LOCAL_MEANS_CACHE`` in per-round setup: this
module-level cache otherwise reuses local means by image-array identity despite
fresh extraction contexts. Both image and support convolutions execute in every
measured round. Cache clearing, ROI preparation and extractor construction stay
outside timing. Inspection found no other module-level or identity-based
radiomics result cache affecting this workload.

Selected filters do not cache output volumes. LoG updates spacing-derived scalar
state in ``apply``; the same geometry is used every round. Gabor's kernel cache is
reset in per-round setup, so operation-only Gabor cases measure fresh kernel
generation even after warmup. The level-2 2D separable wavelet currently
evaluates four rotations regardless of its ``rotation_invariance`` parameter;
its workload metadata records the effective count. Warmup absorbs library
initialization; complete
radiomics keeps image-derived results uncached. These runs do not measure cold
interpreter/import or cold filesystem latency. All allocations and Python object
construction performed by the operation stay included. Garbage collection stays enabled.

Suite version 2 renames complete extraction to ``test_complete_fresh_extraction``
and uses workload IDs ``radiomics/all_fresh/{size}``. Earlier
``test_complete_extraction`` / ``radiomics/all/{size}`` timing results were warmed
by the local-means result cache and are **invalid as fresh-extraction references**.
Do not merge or compare those timing series. Texture/spatial workload identities
are unchanged. Earlier RSS/allocation measurements already used fresh processes
without warmup; their methodology is unaffected, although the shared complete
extraction workload ID is now renamed in memory output too.

Suite version 3 adds named coverage tiers, per-family and aggregation-path
signals, and the 71 registry-driven IBSI workload IDs. It also replaces the
combined ``radiomics/texture/medium`` signal with individual texture-family
measurements. Treat absent version-3 rows in older JSON as unavailable rather
than unchanged performance.

Suite version 4 redesigns the synthetic filtering group around matched paths,
changes Gabor operation timings to fresh-kernel calls, and leaves dedicated
volume-scaling studies out of this group. Old ``filtering/*`` results should
not be compared with version-4 rows by test name: some parameters and timed
cache states changed.
All filtering workloads now record their actual dimensionality, with Gabor
identified as filtering on 2D planes.

Suite version 5 removes the small and large complete fresh-extraction and spatial
statistics timing cases. Their medium-size cases remain. Older timing JSON may contain the four removed
rows; comparison reports treat them as unavailable in the current suite.

Suite version 6 removes the synthetic ``pipeline/log_radiomics/large`` timing case
and includes all ``ibsi1`` cases in ``standard``. The focused ``ibsi`` suite still
runs all 71 published cases; ``exhaustive`` adds ``ibsi2_phase1`` and
``ibsi2_phase2`` to ``standard``. Older timing JSON may lack the newly standard
IBSI I rows or contain the removed pipeline row; comparison reports treat those
differences as unavailable workloads.

Suite version 7 increases IBSI I feature workflows from one to three measured
rounds. Their timed operation and workload IDs are unchanged, but newer timing
JSON contains a preliminary within-run distribution instead of a single sample.

Suite version 8 moves all 151 speed case declarations to a shared case registry
used by timing collection and the memory CLI. ``standard`` selects 100 IDs,
``ibsi`` selects 71, and ``exhaustive`` selects 151 in both runners. Existing
workload IDs and speed measurement regions are unchanged. The older six-case
memory selection and its aliases are removed. Memory setup hooks now run before each operation, matching
the fresh-kernel and fresh-radiomics cache policy used by timing.

Suite version 9 increases IBSI II phase-II feature workflows from one to three
measured rounds. Their operation, setup, validation and workload IDs remain
unchanged; the extra samples provide a preliminary within-run distribution.

SimpleITK and OpenCV have explicit thread APIs. OpenCV GCD builds require
``setNumThreads(0)`` to disable parallel regions; other backends use 1. The
reported serial state is checked, and prior settings are restored afterward. NumPy/SciPy BLAS and OpenMP pools
are limited with ``threadpoolctl`` after imports; SciPy FFT uses ``set_workers(1)``.
The launchers additionally set ``OMP_NUM_THREADS``, ``OPENBLAS_NUM_THREADS``,
``MKL_NUM_THREADS`` and ``ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS`` to 1 before import.
These cover the numerical backends used by the installed dependencies. PyWavelets
and the selected ndimage kernels have no separate thread-pool setting here.
Joblib batch parallelism is not invoked. Thread limits are restored after pytest.

On macOS the launchers also set ``VECLIB_MAXIMUM_THREADS=1``. Apple Accelerate is
not introspected by threadpoolctl, so its limit is recorded as **unverified**;
an empty native-pool list does not prove serial BLAS execution. Direct pytest
users on such systems should export this variable before starting Python. Prefer
the canonical Linux/OpenBLAS environment for controlled comparisons. See
`threadpoolctl's Accelerate limitation <https://github.com/joblib/threadpoolctl/issues/135>`_.

Use **median** as the headline, retaining mean, min/max, standard deviation, IQR,
rounds and iterations in native JSON. ``--benchmark-json`` includes round samples;
use ``--benchmark-save-data`` with saved runs to retain them there too. A minimum
is not a regression verdict. IQR is within-run spread, not a confidence interval
for changes between runs. Rerun noisy changes on an idle machine; compare repeated
runs in both execution orders. Do not infer improvements from differences similar
to normal variability. No timing or memory failure thresholds are installed.

Saving and comparing references
-------------------------------

Native JSON is the sole timing format. No timing database or custom baseline
updater is introduced. For example::

    python tests/benchmarks/run.py --suite standard --benchmark-save=master-accepted --benchmark-save-data
    python tests/benchmarks/run.py --suite standard --benchmark-autosave --benchmark-save-data
    pytest-benchmark list
    # Replace 0001 with the explicit accepted reference ID from the listing
    python tests/benchmarks/run.py --suite standard --benchmark-compare=0001 --benchmark-save=pr-candidate

Or use explicit files (create parent directories first)::

    mkdir -p reports/benchmarks
    python tests/benchmarks/run.py --suite standard --benchmark-json=reports/benchmarks/current.json
    python tests/benchmarks/compare_results.py --current reports/benchmarks/current.json --master reports/benchmarks/master.json --release reports/benchmarks/release.json
    pytest-benchmark compare reports/benchmarks/master.json reports/benchmarks/current.json --columns=median,iqr,mean,stddev,min,max,rounds,iterations

``compare_results.py`` uses explicit reference roles and reports
``100 * (current_median - reference_median) / reference_median`` separately for
master and release. Thus 1 s to 2 s is **+100% regression**, 2 s to 1 s is
**-50% improvement**, and equal values are **0% unchanged**. These labels describe
direction, not statistical significance. Reference/current IQR values are shown
as spread, without interpreting IQR changes as performance improvements.
Workloads are matched by ``extra_info.workload_id`` (or full test name for JSON
without that field); unmatched workloads are reported as unavailable. A zero or
invalid reference timing produces no percentage verdict.

Do not use ``pytest-benchmark compare --between`` for reports of candidate changes:
pytest-benchmark 5.3.0 sorts filenames and may select ``current.json`` as its
baseline, regardless of argument order. Previously generated summaries using that
command must be regenerated from the raw JSON. The raw timing samples are not
changed by the reporting fix; the complete-extraction exception is described above.
The ordinary native comparison command preserves all the statistics. With no explicit reference,
``--benchmark-compare`` uses the latest saved run, which may be a candidate:
**always name the accepted reference when making regression decisions**.

* **Master reference:** accepted master SHA, updated deliberately after accepted
  changes reach master. A slower passing PR cannot redefine it.
* **Release reference:** a specific published tag/SHA, fixed until the next
  release reference is intentionally created. Retain its environment and harness.
* **Current/PR:** a candidate, saved separately. Never overwrite either reference.

Names like ``master-accepted`` are labels, not an automatic branch checkout.
A named save must actually execute the intended revision. Do not run another
checkout's tests while an editable installation still points at the current code.
Long-lived reference JSON belongs in a controlled artifact archive; the repository
ignores generated JSON, ``.benchmarks``, reports and profiling captures.

Same-machine comparisons and CI
-------------------------------

A portable comparison runner resolves refs once, archives each tracked revision
into a fresh temporary source tree, installs it **non-editably** in a separate
virtual environment, and runs a single copy of the **current benchmark harness**
outside those source trees. It asserts that ``zrad.__file__`` belongs to the
intended environment. This avoids stale editable installs, bytecode/build products,
and accidental benchmark-definition differences. Each revision is run sequentially
on the same machine and Python interpreter with identical thread controls.

For local committed changes::

    git fetch origin master --tags
    python tests/benchmarks/compare_revisions.py --master origin/master --release v26.8.0 --current HEAD --output reports/benchmarks/comparison-001

Replace the example release tag with the latest **published release** you intend
to evaluate; the newest version-sorted tag is not necessarily a release. Use
``--suite standard`` for the default comparisons or ``--suite exhaustive``
for the complete published IBSI matrix. ``--full`` is a deprecated alias for
``--suite exhaustive``. Uncommitted production edits are excluded by
``git archive``; use the ordinary launcher to time working-tree edits. The current
working-tree harness is used intentionally. The output directory must be new,
protecting earlier results. Source trees and environments are removed afterward.

Dependencies are resolved separately from each revision's declared requirements.
All use the same benchmark tools, but compatible version ranges can resolve to
different libraries. ``*-dependencies.txt`` and per-run metadata expose this
confounder. Use ``--constraints /absolute/path/constraints.txt`` to supply a common
set when comparing algorithm changes with fixed dependencies. No silently forced
incompatible dependency set or persistent shared editable environment is used.

The separate ``benchmark.yml`` workflow uses **ubuntu-24.04, Python 3.12** on one
GitHub-hosted runner. It runs on PRs/pushes to master and manual dispatch. On PRs,
``HEAD`` is GitHub's tested merge candidate; ``origin/master`` is resolved at job
checkout time. On master pushes, current and master may be identical (a useful
noise observation, not a claimed speedup). The GitHub release API resolves the
latest published stable release, then the runner freezes its SHA for that job.
It does not alter any saved accepted/release reference. Pull requests and pushes
run ``standard``; manual dispatch selects any suite, and a weekly scheduled run uses
``exhaustive``. Exhaustive manual/scheduled runs also create a parallel isolated-
process RSS artifact for all 151 parity-matched cases, one sample each. The
compatibility/coverage workflow remains separate.

Artifacts retained for 30 days contain valid ``current.json``, ``master.json``,
and ``release.json`` where available, installation/test logs, dependency lists,
a status/commit manifest and explicitly reference-relative median/IQR comparison output. The comparison
also appears in the job summary, with current/master and current/release
percentage changes shown separately where each reference is valid. GitHub-hosted hardware and background load vary;
the runner image label does not pin hardware or every OS package. Results are
informative, without small-percentage gates. Measurement/test failures in the
candidate still fail the job. No privileged ``pull_request_target`` or PR-comment
publishing is used.

Old revisions may lack the current API or fail current validation. They are
reported as incompatible/failed, with logs; partial failing results are labelled
``INVALID`` and excluded from comparison. Candidate failures are never silently
skipped. In particular, v26.8.0 does not export ``RieszLoG`` or ``Simoncelli``.
The revision-comparison runner imports the current workload module before suite
selection, so it cannot compare that release without a compatibility adapter.
The initial framework prioritizes PR/master comparisons instead of inventing
historical adapters or claiming unequal workflows are equivalent. Release
tracking starts when a release supports this harness (or a reviewed
common-workload adapter is added). Never interpret an absent row/reference as
unchanged performance.

Memory: separate metrics and processes
--------------------------------------

Peak RSS is measured without pytest or timing instrumentation::

    python tests/benchmarks/memory.py --mode rss --suite standard --repeats 3 --output reports/benchmarks/rss-standard-001.json
    python tests/benchmarks/memory.py --mode rss --suite ibsi --output reports/benchmarks/rss-ibsi-001.json
    python tests/benchmarks/memory.py --mode rss --workload ibsi/i/c/3d/merg --workload ibsi/ii/phase_ii/3.b --output reports/benchmarks/rss-selected-001.json
    python tests/benchmarks/memory.py --mode rss --suite exhaustive --output reports/benchmarks/rss-exhaustive.json

The default is ``standard``. Parity-matched tiers use the same case IDs as
``run.py``: 100 ``standard`` cases, 71 ``ibsi`` cases, and 151 ``exhaustive``
cases. Their inputs, parameters, operation, setup hook, and validator come from
the same case declaration. Join timing ``extra_info.workload_id`` to memory
``measurements[].workload_id`` to inspect both metrics for a case.
Each repetition executes in a **fresh subprocess**, with no warmup. Archive extraction occurs in
the parent before any measured child starts. ``resource.getrusage(RUSAGE_SELF)``
is sampled after computation and before validation/report serialization. Linux
KiB values and macOS byte values are normalized to bytes.

``peak_rss_bytes`` is the **process-lifetime maximum resident set through completion
of the workload**, including interpreter/native imports, loaded inputs and setup.
``setup_peak_rss_bytes`` is the high-water mark immediately before computation;
it is neither current RSS nor a number to subtract to obtain temporary allocation
size. Setup may itself dominate a small operation. The maximum is not polluted by
previous workloads because every sample has its own process/PID. A workload's
setup hook, if any, runs once after input preparation and before the setup RSS
snapshot. Children do not spawn batch workers. A process tree would require a
different measurement design.
The named memory tiers use this same fresh-process method. RSS remains a cold-process
high-water mark, while speed uses an unmeasured warmup and several timed rounds;
equal workload IDs mean equal case definitions, not interchangeable metrics.
Use Memray only for selected cases after RSS screening.
This backend supports Linux/macOS; Windows timing remains usable and optional
profiling dependencies are excluded there.

For allocation attribution, install the optional profiling extra::

    python -m pip install -e '.[test,profiling]'
    python tests/benchmarks/memory.py --mode memray --workload ibsi/ii/phase_ii/3.b --output reports/benchmarks/alloc-ibsi-001.json
    python tests/benchmarks/memory.py --mode memray --native --workload radiomics/spatial/medium --output reports/benchmarks/alloc-spatial-001.json
    python -m memray stats reports/benchmarks/alloc-spatial-001-radiomics-spatial-medium-0.memray
    python -m memray flamegraph reports/benchmarks/alloc-spatial-001-radiomics-spatial-medium-0.memray

Memray's tracker wraps **only one operation after input preparation**, in its own
fresh child. It captures Python and native heap allocations. ``--native`` adds
native stack attribution and extra overhead; it is optional. The reported
``allocation_high_water_bytes`` is Memray's peak simultaneously live **tracked
allocations during the operation**, not peak RSS/RAM. Preexisting input allocations
are not counted. Validation and setup are outside the tracker. No timing JSON is
created and RSS is not reported from an instrumented run. Do not combine those
runtimes with timing results. Native-symbol reports should be generated on the
capture machine; large captures are ignored by git. See
`Memray native tracking <https://bloomberg.github.io/memray/run.html>`_.

pytest-memray was investigated but is not installed: its per-test instrumentation
would include fixture/test work and complicate timing separation. Direct Memray
tracking provides the desired narrower region with one optional dependency.
Memray remains limited to Linux/macOS and is never a normal runtime dependency.

Memory JSON has ``schema_version: 1`` and a ``measurements`` list: stable workload
ID, commit, working-tree dirty flag, PID, timestamp, Python/platform,
workload/environment metadata, ``memory_methodology_version`` (2 for
setup-before-operation), and one mode-specific metric. All repetitions remain
separate; no timing schema is repurposed. Existing output paths are refused.
Peak RSS is suitable for an initial platform-specific longitudinal series;
characterize variance before gating it.
Memray results and captures are diagnostic allocation evidence, not substitute RSS.

Initial observations and extending the suite
--------------------------------------------

Before the suite-version-2 cache correction, local verification on macOS/Apple Silicon,
Python 3.14.6, measured 20 fast
cases in approximately 6.6 seconds and 11 slow cases in approximately 111 seconds
(warmups/setup included in suite duration). IBSI medians were approximately 0.81 s
and 24 s. Earlier six-case fresh-process RSS observations ranged around 300 MiB for resampling,
264 MiB Riesz-LoG, 241 MiB radiomics, 223 MiB spatial, 361 MiB IBSI I and 1.28 GiB
IBSI II. These are development observations, **not versioned baselines**; Accelerate
threading is unverified and dependencies/hardware differ from CI. Recheck on your
machine. First-time ZIP extraction also changes total suite duration, not operation
timing.

When adding a benchmark:

1. Inspect the production path, mutation, retained buffers, caches and threading.
   Select an operation that contributes material workload cost.
2. Add a deterministic factory with a stable workload ID and explicit parameters.
   Construct/load inputs and operators before returning the measured callable.
3. Time only the intended region. Put output checks in the validator. Reuse
   published references where available, without timing reference loading.
4. Make repeated calls independent. Use read-only input arrays where possible;
   for mutating APIs use ``benchmark.pedantic(setup=...)`` to clone inputs outside
   each timed round. Never merely reset once before a multi-iteration round.
5. Add the case and its parameters to ``tests/benchmarks/cases.py``. The named
   tiers then select the same ID in timing and memory. Mark costly cases
   ``benchmark_slow`` and filesystem cases ``benchmark_io`` as appropriate.
6. Include the explicit ``benchmark`` fixture in each timing test so native
   exclusion works. Call ``measure(case.build())`` using the registry case,
   run the test, and inspect the saved JSON.
7. Run the case once in isolated RSS mode and check its setup policy and validator.
   Keep memory instrumentation out of authoritative timing runs.

The current matrix includes selected Laws/Gabor, Simoncelli, and Riesz paths;
the standard tier includes IBSI I features, and the exhaustive tier adds IBSI II
phase-I filters and phase-II features.
Pure NIfTI I/O and joblib batch scaling remain outside this computation-focused
suite. Batch work mixes filesystem and parallel scheduling costs requiring a
separate methodology. The published IBSI workflows provide system-level signals
alongside operation benchmarks.

Useful future work includes a stable dedicated runner, repeated/counterbalanced
revision order, archived environment constraints, characterized per-workload noise
budgets, a reviewed common subset for older releases, and larger or sparse ROI
shapes to exercise spatial algorithm crossover. These are future improvements,
not claims made by the current implementation.
