"""Matched operation-only filtering paths."""

import pytest

from .cases import pytest_params

pytestmark = pytest.mark.benchmark(group='filtering')


@pytest.mark.parametrize('case', pytest_params('test_apply'))
def test_apply(benchmark, measure, case):
    measure(case.build())
