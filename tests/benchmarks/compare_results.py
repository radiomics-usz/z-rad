"""Report explicitly reference-relative changes from native timing JSON files."""

import argparse
import json
import math
import sys
from pathlib import Path

TESTS = Path(__file__).resolve().parents[1]
if str(TESTS) not in sys.path:
    sys.path.insert(0, str(TESTS))


def median_change(reference, current):
    """Classify runtime direction, not statistical significance."""
    if not all(math.isfinite(value) and value >= 0 for value in (reference, current)):
        return 'N/A (invalid timing)'
    if reference == 0:
        return 'N/A (zero reference)'
    percent = 100 * ((current - reference) / reference)
    if percent == 0:
        return '0% unchanged'
    direction = 'regression' if percent > 0 else 'improvement'
    return f'{percent:+.3g}% {direction}'


def workload_results(payload):
    results = {}
    for benchmark in payload['benchmarks']:
        # Versioned workload IDs prevent old cached-extraction results from
        # being matched to fresh extraction even if a test name is reused.
        identifier = benchmark.get('extra_info', {}).get('workload_id') or benchmark['fullname']
        if identifier in results:
            raise ValueError(f'Duplicate workload: {identifier}')
        results[identifier] = benchmark['stats']
    return results


def exhaustive_ibsi_coverage(payload):
    """Return executed/expected published IBSI workload IDs when any are present."""
    from ibsi_cases import ALL_IBSI_PERFORMANCE_CASES

    expected = {case.identifier for case in ALL_IBSI_PERFORMANCE_CASES}
    executed = expected & workload_results(payload).keys()
    return executed, expected


def render_comparison(reference, current, reference_label):
    reference_results = workload_results(reference)
    current_results = workload_results(current)
    table = [
        [
            'Workload',
            f'{reference_label} median (s)',
            'current median (s)',
            'Median change',
            f'{reference_label} IQR (s)',
            'current IQR (s)',
        ]
    ]
    for identifier in sorted(reference_results.keys() | current_results.keys()):
        before = reference_results.get(identifier)
        after = current_results.get(identifier)
        change = (
            median_change(before['median'], after['median'])
            if before is not None and after is not None
            else 'N/A (workload absent from one run)'
        )
        table.append(
            [
                identifier,
                f"{before['median']:.9g}" if before is not None else 'N/A',
                f"{after['median']:.9g}" if after is not None else 'N/A',
                change,
                f"{before['iqr']:.9g}" if before is not None else 'N/A',
                f"{after['iqr']:.9g}" if after is not None else 'N/A',
            ]
        )
    widths = [max(len(cell) for cell in column) for column in zip(*table)]
    rows = [
        f'Current vs {reference_label}',
        f'Median change = 100 * (current - {reference_label}) / {reference_label}.',
        'Positive = regression; negative = improvement; zero = unchanged.',
        'These labels describe direction only, not statistical significance. Inspect IQR and repeat runs.',
        '',
    ]
    executed, expected = exhaustive_ibsi_coverage(current)
    if executed:
        rows.extend([f'Published IBSI performance coverage: {len(executed)}/{len(expected)} cases.', ''])
    for row in table:
        rows.append(' | '.join(cell.ljust(width) for cell, width in zip(row, widths)))
    return '\n'.join(rows) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--current', type=Path, required=True, help='Candidate native timing JSON.')
    parser.add_argument('--master', type=Path, help='Accepted master native timing JSON.')
    parser.add_argument('--release', type=Path, help='Published release native timing JSON.')
    args = parser.parse_args()
    if args.master is None and args.release is None:
        parser.error('Supply --master and/or --release as an explicit reference.')
    current = json.loads(args.current.read_text())
    for label in ('master', 'release'):
        path = getattr(args, label)
        if path is not None:
            print(render_comparison(json.loads(path.read_text()), current, label))


if __name__ == '__main__':
    main()
