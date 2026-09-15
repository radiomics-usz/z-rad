"""Safety contracts for performance measurements, without measuring performance."""

import json
from types import SimpleNamespace

import pytest
from benchmarks.cases import ALL_CASES, CASE_BY_ID, pytest_params, suite_cases
from benchmarks.compare_results import exhaustive_ibsi_coverage, median_change, render_comparison
from benchmarks.compare_revisions import write_summary
from benchmarks.conftest import measure, pytest_configure
from benchmarks.memory import MEMORY_SUITES, exhaustive_names
from benchmarks.suites import SUITE_MARKERS, marker_for_suite
from ibsi_cases import ALL_IBSI_PERFORMANCE_CASES, IBSI_I_FEATURE_CASES, IBSI_II_FEATURE_CASES, IBSI_II_FILTER_CASES

pytestmark = pytest.mark.unit


def test_exhaustive_ibsi_registry_is_complete_and_unique():
    assert len(IBSI_I_FEATURE_CASES) == 20
    assert len(IBSI_II_FILTER_CASES) == 33
    assert len(IBSI_II_FEATURE_CASES) == 18
    identifiers = [case.identifier for case in ALL_IBSI_PERFORMANCE_CASES]
    assert len(identifiers) == len(set(identifiers)) == 71
    assert {case.config for case in IBSI_II_FEATURE_CASES} == {
        f'{number}.{variant}' for number in range(1, 10) for variant in 'AB'
    }


def test_named_suite_markers_keep_ibsi_phase_one_and_two_opt_in():
    assert marker_for_suite('standard') == 'not benchmark_exhaustive'
    assert marker_for_suite('ibsi') == 'benchmark_ibsi'
    assert marker_for_suite('exhaustive') is None
    with pytest.raises(ValueError, match='Unknown benchmark suite'):
        marker_for_suite('typo')


def test_exhaustive_coverage_reports_registry_cases():
    workload_id = ALL_IBSI_PERFORMANCE_CASES[0].identifier
    executed, expected = exhaustive_ibsi_coverage(timing_result(1.0, workload_id))
    assert executed == {workload_id}
    assert len(expected) == 71


def test_exhaustive_memory_inventory_includes_every_registry_case():
    names = exhaustive_names()
    assert len(names) == len(set(names)) == 151
    assert {case.identifier for case in ALL_IBSI_PERFORMANCE_CASES} < set(names)


def test_memory_and_timing_case_inventory_has_named_tier_parity():
    assert len(ALL_CASES) == len(CASE_BY_ID) == 151
    assert set(MEMORY_SUITES) == set(SUITE_MARKERS) == {'standard', 'ibsi', 'exhaustive'}
    for tier, count in [('standard', 100), ('ibsi', 71), ('exhaustive', 151)]:
        cases = suite_cases(tier)
        assert len(cases) == count
        assert len({case.identifier for case in cases}) == count
    assert {case.identifier for case in suite_cases('exhaustive')} == set(exhaustive_names())
    for test in {case.test for case in ALL_CASES if case.pytest_id}:
        assert {param.values[0].identifier for param in pytest_params(test)} == {
            case.identifier for case in ALL_CASES if case.test == test
        }


def timing_config(**overrides):
    options = {
        'benchmark_only': True,
        'numprocesses': 0,
        'no_cov': True,
        'memray': False,
        'benchmark_cprofile': None,
        'benchmark_disable_gc': False,
        **overrides,
    }
    return SimpleNamespace(getoption=lambda name, default=None: options.get(name, default))


@pytest.mark.parametrize(
    ('overrides', 'message'),
    [
        ({'numprocesses': 'auto'}, '-n 0'),
        ({'no_cov': False}, '--no-cov'),
        ({'memray': True}, 'no memory/CPU profilers'),
        ({'benchmark_cprofile': 'tottime'}, 'no memory/CPU profilers'),
        ({'benchmark_disable_gc': True}, 'garbage collection enabled'),
    ],
)
def test_timing_rejects_contaminated_measurements(overrides, message):
    with pytest.raises(pytest.UsageError, match=message):
        pytest_configure(timing_config(**overrides))


def test_timing_rejects_an_xdist_worker_even_if_worker_option_is_zero():
    config = timing_config()
    config.workerinput = {}
    with pytest.raises(pytest.UsageError, match='-n 0'):
        pytest_configure(config)


def test_timing_accepts_serial_uninstrumented_configuration():
    pytest_configure(timing_config())


def test_normal_correctness_tests_keep_coverage_and_workers():
    pytest_configure(timing_config(benchmark_only=False, numprocesses='auto', no_cov=False))


def test_native_thread_controls_are_serial_and_restore_previous_settings():
    import cv2
    import SimpleITK as sitk
    from benchmarks.runtime import environment_metadata, single_threaded

    before = (cv2.getNumThreads(), sitk.ProcessObject.GetGlobalDefaultNumberOfThreads())
    with single_threaded():
        info = environment_metadata()
        assert info['opencv_threads'] == info['itk_threads'] == info['scipy_fft_workers'] == 1
    assert (cv2.getNumThreads(), sitk.ProcessObject.GetGlobalDefaultNumberOfThreads()) == before


def timing_result(median, workload_id='example/workload'):
    return {
        'benchmarks': [
            {
                'fullname': 'tests/benchmarks/test_example.py::test_example',
                'extra_info': {'workload_id': workload_id},
                'stats': {'median': median, 'iqr': 0.1},
            }
        ],
    }


@pytest.mark.parametrize('reference_label', ['master', 'release'])
@pytest.mark.parametrize(
    ('reference', 'current', 'expected'),
    [(1.0, 2.0, '+100% regression'), (2.0, 1.0, '-50% improvement'), (1.0, 1.0, '0% unchanged')],
)
def test_summary_changes_are_reference_relative(tmp_path, reference_label, reference, current, expected):
    # current.json sorts before both reference names. Argument/filename order
    # must never choose the arithmetic baseline.
    states = {}
    for label, value in [(reference_label, reference), ('current', current)]:
        path = tmp_path / f'{label}.json'
        path.write_text(json.dumps(timing_result(value)))
        states[label] = {'status': 'ok', 'commit': label, 'result': str(path)}
    write_summary(tmp_path, states)
    report = (tmp_path / 'comparison.txt').read_text()
    assert f'Current vs {reference_label}' in report
    assert expected in report
    row = next(line for line in report.splitlines() if line.startswith('example/workload'))
    assert row.split('|')[3].strip() == expected


def test_summary_uses_each_references_own_value(tmp_path):
    states = {}
    for label, value in [('master', 1.0), ('release', 4.0), ('current', 2.0)]:
        path = tmp_path / f'{label}.json'
        path.write_text(json.dumps(timing_result(value)))
        states[label] = {'status': 'ok', 'commit': label, 'result': str(path)}
    write_summary(tmp_path, states)
    report = (tmp_path / 'comparison.txt').read_text()
    master, release = report.split('Current vs release')
    assert '+100% regression' in master
    assert '-50% improvement' in release


@pytest.mark.parametrize('reference', [0.0, -1.0, float('nan'), float('inf')])
def test_invalid_reference_does_not_produce_a_performance_verdict(reference):
    assert median_change(reference, 1.0).startswith('N/A')


def test_fresh_extraction_is_not_compared_to_cached_results_with_the_same_test_name():
    report = render_comparison(
        timing_result(1.0, 'radiomics/all/small'),
        timing_result(2.0, 'radiomics/all_fresh/small'),
        'master',
    )
    assert report.count('N/A (workload absent from one run)') == 2
    assert '% regression' not in report and '% improvement' not in report


def test_fresh_extraction_recomputes_local_means_in_every_round(monkeypatch):
    from benchmarks.runtime import single_threaded
    from benchmarks.workloads import radiomics

    import zrad.radiomics.intensity as intensity

    timed = False
    setup_calls = []
    convolution_counts = []
    cache_hits = []

    class GuardedCache(dict):
        def clear(self):
            assert not timed, 'Cache reset must be outside the measured operation'
            setup_calls.append(True)
            super().clear()

    cache = GuardedCache()
    monkeypatch.setattr(intensity, '_LOCAL_MEANS_CACHE', cache)
    workload = radiomics('small')
    with single_threaded():
        # Seed a real cached result before exercising the benchmark lifecycle.
        workload.validate(workload.operation())
        assert cache
        original_convolve = intensity.convolve
        original_lookup = intensity._get_cached_local_means

        def convolve(*args, **kwargs):
            assert timed
            convolution_counts[-1] += 1
            return original_convolve(*args, **kwargs)

        def lookup(*args, **kwargs):
            result = original_lookup(*args, **kwargs)
            cache_hits.append(result is not None)
            return result

        monkeypatch.setattr(intensity, 'convolve', convolve)
        monkeypatch.setattr(intensity, '_get_cached_local_means', lookup)

        class UntimedBenchmark:
            extra_info = {}

            def pedantic(self, target, *, setup=None, rounds, iterations, warmup_rounds):
                nonlocal timed
                assert iterations == 1
                # Exercise the fixture's real workload/setup wiring without
                # recording performance or requiring the benchmark fixture.
                for _ in range(warmup_rounds + rounds):
                    if setup is not None:
                        setup()
                    convolution_counts.append(0)
                    timed = True
                    try:
                        result = target()
                    finally:
                        timed = False
                return result

        measure.__wrapped__(UntimedBenchmark(), {})(workload)

    assert len(setup_calls) == workload.rounds + 1
    assert convolution_counts == [2] * (workload.rounds + 1)
    assert cache_hits == [False] * (workload.rounds + 1)
    assert workload.identifier == 'radiomics/all_fresh/small'


def test_gabor_operation_rebuilds_kernels_after_each_round_setup():
    from benchmarks.workloads import filtering

    workload = filtering('gabor_fixed', 'small')
    flt = workload.operation.func.__self__
    assert workload.setup is not None

    workload.setup()
    workload.validate(workload.operation())
    first = flt._make_kernels.cache_info()
    assert first.misses == 1 and first.hits == 0

    workload.operation()
    assert flt._make_kernels.cache_info().hits == 1

    workload.setup()
    workload.validate(workload.operation())
    second = flt._make_kernels.cache_info()
    assert second.misses == 1 and second.hits == 0
