"""Run one current harness against isolated installed revisions on this machine.

No editable installs, checkout switching, persistent reference replacement, or
benchmark result database. Outputs are native pytest-benchmark JSON and logs.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import venv
from pathlib import Path

if __package__:
    from .compare_results import render_comparison
    from .runtime import controlled_environment
    from .suites import SUITE_MARKERS, marker_for_suite
else:
    from compare_results import render_comparison
    from runtime import controlled_environment
    from suites import SUITE_MARKERS, marker_for_suite

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tests'))
TOOLS = ['pytest', 'pytest-cov', 'pytest-xdist', 'pytest-benchmark>=5.3.0', 'threadpoolctl>=3.5']


def expected_exhaustive_ids():
    from ibsi_cases import ALL_IBSI_PERFORMANCE_CASES

    return {case.identifier for case in ALL_IBSI_PERFORMANCE_CASES}


def resolve_revision(repository, ref):
    # --end-of-options prevents a supplied ref from being interpreted as an option.
    return subprocess.check_output(
        ['git', 'rev-parse', '--verify', '--end-of-options', ref + '^{commit}'], cwd=repository, text=True
    ).strip()


def copy_harness(destination):
    tests = destination / 'tests'
    tests.mkdir(parents=True)
    for name in ('conftest.py', 'ibsi_cases.py', 'ibsi_helpers.py'):
        shutil.copy2(ROOT / 'tests' / name, tests / name)
    ignore = shutil.ignore_patterns('__pycache__', '.cache', 'IBSI_SUV', '*.pyc')
    for name in ('benchmarks', 'data'):
        shutil.copytree(ROOT / 'tests' / name, tests / name, ignore=ignore)
    shutil.copy2(ROOT / 'pytest.ini', destination / 'pytest.ini')


def run_revision(label, sha, args, temporary, harness):
    source = temporary / label / 'source'
    source.mkdir(parents=True)
    archive = source.parent / 'source.tar'
    subprocess.run(['git', 'archive', '--format=tar', '-o', str(archive), sha], cwd=args.repository, check=True)
    with tarfile.open(archive) as bundle:
        bundle.extractall(source, filter='data')
    archive.unlink()
    env_dir = source.parent / 'venv'
    venv.EnvBuilder(with_pip=True).create(env_dir)
    python = env_dir / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    environment = controlled_environment()
    # Never inherit an editable checkout or user-supplied instrumentation.
    for key in ('PYTHONPATH', 'PYTHONHOME', 'PYTEST_ADDOPTS', 'COVERAGE_PROCESS_START', 'COVERAGE_RCFILE'):
        environment.pop(key, None)
    environment.update(PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1', ZRAD_BENCHMARK_COMMIT=sha)
    result_path = args.output / f'{label}.json'
    with (args.output / f'{label}.log').open('w') as log:

        def run(command):
            return subprocess.run(
                command, cwd=harness, env=environment, stdout=log, stderr=subprocess.STDOUT, check=False
            ).returncode

        # No [test] extra from the historical revision: test tooling is the same
        # current tool set, installed alongside each revision's own dependencies.
        install = [str(python), '-m', 'pip', 'install', str(source), *TOOLS]
        if args.constraints:
            install.extend(['-c', str(args.constraints)])
        if run(install) or run([str(python), '-m', 'pip', 'check']):
            return {'status': 'installation_failed', 'commit': sha}
        with (args.output / f'{label}-dependencies.txt').open('w') as freeze:
            subprocess.run(
                [str(python), '-m', 'pip', 'freeze'], env=environment, cwd=harness, stdout=freeze, check=True
            )
        # Assert the measured code really comes from THIS revision's environment.
        probe = (
            'from pathlib import Path; import sys, zrad; '
            'p=Path(zrad.__file__).resolve(); print(p, flush=True); '
            'assert p.is_relative_to(Path(sys.prefix).resolve()); '
            'sys.path.insert(0, "tests"); import benchmarks.workloads'
        )
        if run([str(python), '-c', probe]):
            return {'status': 'incompatible_harness_or_import', 'commit': sha}
        command = [
            str(python),
            '-m',
            'pytest',
            'tests/benchmarks',
            '--benchmark-only',
            '-n',
            '0',
            '--no-cov',
            '--benchmark-json=' + str(result_path),
            '--benchmark-columns=median,iqr,mean,stddev,min,max,rounds,iterations',
        ]
        marker = marker_for_suite(args.suite)
        if marker:
            command.extend(['-m', marker])
        if run(command):
            # pytest can write partial results on failures. Never compare them.
            if result_path.exists():
                result_path.rename(args.output / f'{label}-INVALID.json')
            return {'status': 'benchmark_failed', 'commit': sha}
    payload = json.loads(result_path.read_text())
    if not payload.get('benchmarks'):
        return {'status': 'no_measurements', 'commit': sha}
    if args.suite == 'exhaustive':
        measured = {
            benchmark.get('extra_info', {}).get('workload_id') or benchmark['fullname']
            for benchmark in payload['benchmarks']
        }
        missing = sorted(expected_exhaustive_ids() - measured)
        if missing:
            result_path.rename(args.output / f'{label}-INCOMPLETE.json')
            return {'status': 'incomplete_exhaustive_suite', 'commit': sha, 'missing': missing}
    return {'status': 'ok', 'commit': sha, 'python': str(python), 'result': str(result_path)}


def write_summary(output, states):
    rows = [
        'Same-machine performance comparison',
        '===================================',
        '',
        'Median runtime is the headline; inspect IQR and repeated runs before drawing conclusions.',
        'No performance thresholds are enforced. Dependency differences confound algorithm comparisons.',
        '',
    ]
    for label, state in states.items():
        rows.append(f'{label}: {state["commit"]} — {state["status"]} (see {label}.log)')
    rows.append('')
    current = states.get('current', {})
    comparisons = 0
    for label in ('master', 'release'):
        reference = states.get(label, {})
        if current.get('status') != 'ok' or reference.get('status') != 'ok':
            continue
        rows.append(
            render_comparison(
                json.loads(Path(reference['result']).read_text()),
                json.loads(Path(current['result']).read_text()),
                label,
            )
        )
        comparisons += 1
    if not comparisons:
        rows.append('Fewer than two valid runs: no comparison is available.')
    (output / 'comparison.txt').write_text('\n'.join(rows) + '\n')
    # Keep environment paths in the manifest for provenance even after cleanup.
    (output / 'manifest.json').write_text(json.dumps(states, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository', type=Path, default=ROOT)
    parser.add_argument('--current', default='HEAD', help='Committed candidate ref; excludes uncommitted edits.')
    parser.add_argument('--master', default='origin/master', help='Accepted branch ref, resolved once at startup.')
    parser.add_argument('--release', help='Explicit release tag (no implicit rolling baseline).')
    parser.add_argument(
        '--output', type=Path, required=True, help='New result directory; existing directories refused.'
    )
    parser.add_argument(
        '--constraints', type=Path, help='Optional shared pip constraints for controlled dependency versions.'
    )
    parser.add_argument('--suite', choices=SUITE_MARKERS)
    parser.add_argument('--full', action='store_true', help='Deprecated alias for --suite exhaustive.')
    args = parser.parse_args()
    if args.full:
        if args.suite is not None:
            parser.error('--full cannot be combined with --suite; use --suite exhaustive.')
        args.suite = 'exhaustive'
    elif args.suite is None:
        args.suite = 'standard'
    args.repository = args.repository.resolve()
    args.output = args.output.resolve()
    if args.constraints:
        args.constraints = args.constraints.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    refs = ({'release': args.release} if args.release else {}) | {'master': args.master, 'current': args.current}
    refs = {label: resolve_revision(args.repository, ref) for label, ref in refs.items()}
    states = {}
    with tempfile.TemporaryDirectory(prefix='zrad-revisions-') as directory:
        temporary = Path(directory)
        harness = temporary / 'harness'
        copy_harness(harness)
        for label, sha in refs.items():
            print(f'Running {label} ({sha}); log: {args.output / (label + ".log")}', flush=True)
            states[label] = run_revision(label, sha, args, temporary, harness)
            print(f'{label}: {states[label]["status"]}', flush=True)
        write_summary(args.output, states)
    print((args.output / 'comparison.txt').read_text())
    if states['current']['status'] != 'ok':
        raise SystemExit('Candidate benchmark failed; see current.log.')


if __name__ == '__main__':
    main()
