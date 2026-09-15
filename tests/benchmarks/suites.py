"""Named benchmark-suite selection shared by launchers and tests."""

SUITE_MARKERS = {
    'standard': 'not benchmark_exhaustive',
    'ibsi': 'benchmark_ibsi',
    'exhaustive': None,
}


def marker_for_suite(name):
    try:
        return SUITE_MARKERS[name]
    except KeyError as error:
        raise ValueError(f'Unknown benchmark suite {name!r}; choose from {tuple(SUITE_MARKERS)}') from error
