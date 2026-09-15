"""Isolated process RSS or operation-only Memray allocation diagnostics.

Run as a script. No pytest timing fixture, coverage, or xdist is involved.
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

# The same harness also runs outside a checkout against installed revision wheels.
TESTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TESTS))
sys.path.insert(0, str(TESTS.parent))

from benchmarks.cases import CASE_BY_ID, suite_cases  # noqa: E402
from benchmarks.runtime import controlled_environment, environment_metadata, single_threaded  # noqa: E402

MEMORY_SUITES = ('standard', 'ibsi', 'exhaustive')


def exhaustive_names():
    return tuple(case.identifier for case in suite_cases('exhaustive'))


def build_workload(name):
    case = CASE_BY_ID[name]
    if not name.startswith('ibsi/'):
        return case.build()
    from benchmarks.ibsi_workloads import load_ct_sources, load_ibsi_i_digital, load_ibsi_ii_phantoms

    cache = TESTS / 'data/.cache'
    if case.group == 'ibsi2_phase1':
        phantoms = load_ibsi_ii_phantoms(cache / 'ibsi_2_digital_phantom')
        return case.build(phantoms, cache / 'ibsi_2_response_maps')
    sources = {}
    reference = case.reference
    if reference.image_source.startswith('ct_') or reference.mask_source.startswith('ct_'):
        sources.update(load_ct_sources(cache / 'ibsi_ct_radiomics_phantom'))
    if reference.image_source == 'i_digital':
        sources.update(load_ibsi_i_digital(cache / 'ibsi_1_digital_phantom'))
    return case.build(sources)


def peak_rss_bytes():
    import resource

    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == 'darwin' else value * 1024)


def measure_child(args):
    # Load numerical backends before installing runtime thread-pool limits.
    from benchmarks import workloads  # noqa: F401

    if sys.gettrace() or sys.getprofile():
        raise RuntimeError('Run memory measurement without external tracing/profiling.')
    with single_threaded():
        workload = build_workload(args.child)
        info = environment_metadata()
        if workload.setup is not None:
            workload.setup()
        before = peak_rss_bytes()
        if args.mode == 'memray':
            import memray

            capture = args.output.with_suffix('.memray')
            with memray.Tracker(str(capture), native_traces=args.native):
                result = workload.operation()
            metric = {
                'allocation_high_water_bytes': memray.FileReader(str(capture)).metadata.peak_memory,
                'capture': str(capture),
                'native_traces': args.native,
            }
        else:
            result = workload.operation()
            # Snapshot before validation/report serialization to avoid their allocations.
            metric = {'peak_rss_bytes': peak_rss_bytes(), 'setup_peak_rss_bytes': before}
        workload.validate(result)
    return {
        'workload_id': workload.identifier,
        'mode': args.mode,
        'pid': os.getpid(),
        'commit': args.commit,
        'working_tree_dirty': args.dirty,
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'python': platform.python_version(),
        'platform': platform.platform(),
        'metadata': {**info, **workload.metadata},
        'memory_methodology_version': 2,
        **metric,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['rss', 'memray'], default='rss')
    parser.add_argument('--workload', action='append', dest='workloads')
    parser.add_argument('--suite', choices=MEMORY_SUITES)
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--native', action='store_true', help='Native allocation stacks (Memray only).')
    parser.add_argument('--output', type=Path, default=Path('reports/benchmarks/memory.json'))
    parser.add_argument('--child', help=argparse.SUPPRESS)
    parser.add_argument('--commit', help=argparse.SUPPRESS)
    parser.add_argument('--dirty', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if sys.platform not in ('linux', 'darwin'):
        parser.error('This RSS backend supports Linux/macOS only; timing benchmarks also support Windows.')
    if args.repeats < 1 or (args.native and args.mode != 'memray'):
        parser.error('Use positive repeats and --native only with --mode memray.')
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.child:
        args.output.write_text(json.dumps(measure_child(args), indent=2) + '\n')
        return
    if args.output.exists():
        parser.error('Output exists; choose a new path to preserve earlier measurements.')
    available = set(exhaustive_names())
    if args.workloads and args.suite:
        parser.error('Use either --suite or explicit --workload selections.')
    selected_suite = tuple(case.identifier for case in suite_cases(args.suite or 'standard'))
    names = list(args.workloads or selected_suite)
    unknown = sorted(set(names) - available)
    if unknown:
        parser.error('Unknown workload(s): ' + ', '.join(unknown))
    if any(name.startswith(('ibsi/i/', 'ibsi/ii/phase_ii/')) for name in names):
        from conftest import _prepare_data_dir

        cache = TESTS / 'data/.cache'
        cache.mkdir(exist_ok=True)
        _prepare_data_dir(TESTS / 'data/ibsi_ct_radiomics_phantom.zip', cache / 'ibsi_ct_radiomics_phantom')
        if any(name.startswith('ibsi/i/') for name in names):
            _prepare_data_dir(TESTS / 'data/ibsi_1_digital_phantom.zip', cache / 'ibsi_1_digital_phantom')
    if any(name.startswith('ibsi/ii/phase_i/') for name in names):
        from conftest import _prepare_data_dir

        cache = TESTS / 'data/.cache'
        cache.mkdir(exist_ok=True)
        _prepare_data_dir(TESTS / 'data/ibsi_2_digital_phantom.zip', cache / 'ibsi_2_digital_phantom')
        _prepare_data_dir(TESTS / 'data/ibsi_2_reference_data/ibsi_2_response_maps.zip', cache / 'ibsi_2_response_maps')
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=TESTS.parent, capture_output=True, text=True, check=False)
    commit = os.environ.get('ZRAD_BENCHMARK_COMMIT', commit.stdout.strip() or 'unknown')
    status = subprocess.run(
        ['git', 'status', '--porcelain'], cwd=TESTS.parent, capture_output=True, text=True, check=False
    )
    dirty = bool(status.stdout.strip()) and 'ZRAD_BENCHMARK_COMMIT' not in os.environ
    results = []
    # Each subprocess exec has its own lifetime high-water mark, including imports
    # and preparation. ZIP extraction above never enters the measured process.
    with tempfile.TemporaryDirectory(prefix='zrad-memory-') as temporary:
        for name in names:
            for repeat in range(args.repeats):
                safe_name = name.replace('/', '-')
                output = (args.output.parent if args.mode == 'memray' else Path(temporary)) / (
                    f'{args.output.stem}-{safe_name}-{repeat}.json'
                )
                command = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    '--child',
                    name,
                    '--mode',
                    args.mode,
                    '--output',
                    str(output),
                    '--commit',
                    commit,
                ]
                if args.native:
                    command.append('--native')
                if dirty:
                    command.append('--dirty')
                print(f'{args.mode}: {name}, independent process {repeat + 1}/{args.repeats}', flush=True)
                subprocess.run(command, check=True, env=controlled_environment(), cwd=TESTS.parent)
                results.append(json.loads(output.read_text()))
                if args.mode == 'memray':
                    output.unlink()  # capture remains; combined JSON is the manifest
    args.output.write_text(json.dumps({'schema_version': 1, 'measurements': results}, indent=2) + '\n')
    print(args.output)


if __name__ == '__main__':
    main()
