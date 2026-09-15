"""Native benchmark exclusion handles collection; this only guards measurement."""

import gc
import os
import sys
from pathlib import Path

import pytest

from .cases import CASE_BY_ID, single_case, suite_cases
from .runtime import environment_metadata, single_threaded

_BENCHMARK_GROUP_ORDER = {
    'image': 0,
    'preprocessing': 1,
    'filtering': 2,
    'radiomics': 3,
    'ibsi1': 4,
    'ibsi2_phase1': 5,
    'ibsi2_phase2': 6,
}
_BENCHMARK_ITEM_ORDER = {}


def pytest_collection_modifyitems(session, config, items):
    """Keep benchmark output in a stable, workload-oriented order."""

    def sort_key(item):
        marker = item.get_closest_marker('benchmark')
        group = marker.kwargs.get('group') if marker else None
        return _BENCHMARK_GROUP_ORDER.get(group, len(_BENCHMARK_GROUP_ORDER))

    items.sort(key=sort_key)
    _BENCHMARK_ITEM_ORDER.clear()
    for index, item in enumerate(items):
        _BENCHMARK_ITEM_ORDER[item.nodeid] = index
        _BENCHMARK_ITEM_ORDER[item.name] = index


def pytest_collection_finish(session):
    """Fail full named-suite collection if timing and memory IDs diverge."""
    config = session.config
    if not config.getoption('benchmark_only'):
        return
    benchmark_dir = Path(__file__).resolve().parent
    ids = []
    for item in session.items:
        if Path(item.path).parent != benchmark_dir:
            continue
        test = item.originalname or item.name.split('[', 1)[0]
        case = item.callspec.params.get('case') if hasattr(item, 'callspec') else single_case(test)
        if case is None or case.test != test or CASE_BY_ID.get(case.identifier) != case:
            raise pytest.UsageError(f'Unregistered benchmark case: {item.nodeid}')
        group_marker = item.get_closest_marker('benchmark')
        group = group_marker.kwargs.get('group') if group_marker else None
        if (
            group != case.group
            or bool(item.get_closest_marker('benchmark_exhaustive')) != case.exhaustive
            or bool(item.get_closest_marker('benchmark_slow')) != case.slow
            or bool(item.get_closest_marker('benchmark_ibsi')) != case.group.startswith('ibsi')
        ):
            raise pytest.UsageError(f'Benchmark tier/group differs from registry: {item.nodeid}')
        ids.append(case.identifier)
    if len(ids) != len(set(ids)):
        raise pytest.UsageError('Timing collection has duplicate benchmark workload IDs.')
    if len(config.args) != 1 or Path(config.args[0]).resolve() != benchmark_dir:
        return  # Focused native pytest selections need not collect a full tier.
    marker = config.getoption('markexpr')
    tier = {
        '': 'exhaustive',
        'not benchmark_exhaustive': 'standard',
        'benchmark_ibsi': 'ibsi',
    }.get(marker)
    if tier is None:
        return
    expected = {case.identifier for case in suite_cases(tier)}
    if set(ids) != expected:
        missing = sorted(expected - set(ids))
        extra = sorted(set(ids) - expected)
        raise pytest.UsageError(f'{tier} timing/memory inventory differs: missing={missing}, extra={extra}.')


@pytest.hookimpl(wrapper=True)
def pytest_benchmark_group_stats(config, benchmarks, group_by):
    """Keep the console report aligned with benchmark collection order."""

    outcome = yield
    if group_by != 'group':
        return

    groups = list(outcome.get_result() if hasattr(outcome, 'get_result') else outcome)
    fallback = len(_BENCHMARK_ITEM_ORDER)
    for _, grouped_benchmarks in groups:
        grouped_benchmarks.sort(
            key=lambda benchmark: _BENCHMARK_ITEM_ORDER.get(
                benchmark.get('fullname') or benchmark.get('name'), fallback
            )
        )
    groups.sort(key=lambda pair: _BENCHMARK_GROUP_ORDER.get(pair[0], len(_BENCHMARK_GROUP_ORDER)))
    if hasattr(outcome, 'force_result'):
        outcome.force_result(groups)
    else:
        return groups


def pytest_configure(config):
    if not config.getoption('benchmark_only'):
        return
    errors = []
    if config.getoption('numprocesses', default=0) not in (0, None) or hasattr(config, 'workerinput'):
        errors.append('-n 0')
    if not config.getoption('no_cov', default=False):
        errors.append('--no-cov')
    if config.getoption('memray', default=False) or config.getoption('benchmark_cprofile', default=None):
        errors.append('no memory/CPU profilers (use memory.py separately)')
    if config.getoption('benchmark_disable_gc'):
        errors.append('garbage collection enabled')
    if errors:
        raise pytest.UsageError('Authoritative benchmarks require: ' + ', '.join(errors))


@pytest.fixture(scope='session')
def timing_environment(request):
    if not request.config.getoption('benchmark_only'):
        pytest.fail('Use --benchmark-only -n 0 --no-cov for intentional timing runs.')
    if sys.gettrace() or sys.getprofile() or not gc.isenabled():
        pytest.fail('Timing requires no tracing/profiling and enabled garbage collection.')
    with single_threaded():
        yield environment_metadata()


@pytest.fixture
def measure(benchmark, timing_environment):
    """Set up outside measurement; no output/result cache is shared across calls."""

    def run(workload):
        benchmark.extra_info.update(timing_environment)
        benchmark.extra_info.update(workload.metadata)
        benchmark.extra_info['workload_id'] = workload.identifier
        benchmark.extra_info['target_commit'] = os.environ.get('ZRAD_BENCHMARK_COMMIT')
        benchmark.extra_info['measurement'] = 'timing_uninstrumented'
        result = benchmark.pedantic(
            workload.operation, setup=workload.setup, rounds=workload.rounds, iterations=1, warmup_rounds=1
        )
        workload.validate(result)
        return result

    return run


@pytest.fixture(scope='session')
def ibsi_ct_sources(ibsi_ct_data_dir):
    from .ibsi_workloads import load_ct_sources

    return load_ct_sources(ibsi_ct_data_dir)


@pytest.fixture(scope='session')
def ibsi_feature_sources(ibsi_ct_sources, ibsi_i_digital_data_dir):
    from .ibsi_workloads import load_ibsi_i_digital

    return {**ibsi_ct_sources, **load_ibsi_i_digital(ibsi_i_digital_data_dir)}


@pytest.fixture(scope='session')
def ibsi_ii_filter_sources(ibsi_ii_digital_data_dir):
    from .ibsi_workloads import load_ibsi_ii_phantoms

    return load_ibsi_ii_phantoms(ibsi_ii_digital_data_dir)


def pytest_benchmark_update_commit_info(config, commit_info):
    target = os.environ.get('ZRAD_BENCHMARK_COMMIT')
    if target:
        # The external harness is not a git checkout. Identify the installed
        # revision, not a harness directory or a developer's editable package.
        commit_info.update(id=target, dirty=False, project='z-rad', branch=None)
