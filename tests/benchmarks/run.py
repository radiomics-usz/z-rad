"""Convenient native pytest launcher with thread environment set before imports."""

import argparse
import subprocess
import sys
from pathlib import Path

from runtime import controlled_environment
from suites import SUITE_MARKERS, marker_for_suite


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--suite', choices=SUITE_MARKERS)
    options, pytest_args = parser.parse_known_args()
    command = [
        sys.executable,
        '-m',
        'pytest',
        'tests/benchmarks',
        '--benchmark-only',
        '-n',
        '0',
        '--no-cov',
        '--benchmark-columns=median,iqr,mean,stddev,min,max,rounds,iterations',
        *pytest_args,
    ]
    marker = marker_for_suite(options.suite or 'standard')
    if marker:
        if '-m' in pytest_args:
            parser.error('Use either --suite or a native pytest marker expression, not both.')
        command.extend(['-m', marker])
    raise SystemExit(subprocess.call(command, cwd=Path(__file__).resolve().parents[2], env=controlled_environment()))


if __name__ == '__main__':
    main()
