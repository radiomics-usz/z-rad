"""Shared, explicit execution environment for timing and isolated memory runs."""

import hashlib
import os
import sys
from contextlib import contextmanager
from importlib.metadata import distributions
from pathlib import Path

# Applied before native libraries import by the launchers. Runtime APIs below
# also handle ordinary pytest invocations where other tests imported them first.
THREAD_ENV = {
    'OMP_NUM_THREADS': '1',
    'OPENBLAS_NUM_THREADS': '1',
    'MKL_NUM_THREADS': '1',
    'ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS': '1',
}
if sys.platform == 'darwin':
    THREAD_ENV['VECLIB_MAXIMUM_THREADS'] = '1'


def controlled_environment():
    return {**os.environ, **THREAD_ENV}


@contextmanager
def single_threaded():
    import cv2
    import SimpleITK as sitk
    from scipy import fft
    from threadpoolctl import threadpool_limits

    itk_threads = sitk.ProcessObject.GetGlobalDefaultNumberOfThreads()
    cv_threads = cv2.getNumThreads()
    gcd = any('Parallel framework:' in line and 'GCD' in line for line in cv2.getBuildInformation().splitlines())
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    # GCD only supports nonpositive settings; zero disables parallel regions.
    cv2.setNumThreads(0 if gcd else 1)
    try:
        with threadpool_limits(limits=1), fft.set_workers(1):
            yield
    finally:
        sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(itk_threads)
        cv2.setNumThreads((0 if cv_threads == 1 else -1) if gcd else cv_threads)


def environment_metadata():
    import cv2
    import numpy as np
    import SimpleITK as sitk
    from scipy import fft
    from threadpoolctl import threadpool_info

    import zrad

    suite = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(suite.glob('*.py')):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    blas = getattr(np.__config__, 'CONFIG', {}).get('Build Dependencies', {}).get('blas', {})
    limitations = []
    if blas.get('name') == 'accelerate':
        limitations.append('Apple Accelerate is not introspected by threadpoolctl; environment limit is unverified.')
    pools = threadpool_info()
    if any(pool['num_threads'] != 1 for pool in pools):
        raise RuntimeError(f'Native thread pools were not limited to one thread: {pools}')
    if sitk.ProcessObject.GetGlobalDefaultNumberOfThreads() != 1 or cv2.getNumThreads() != 1:
        raise RuntimeError('SimpleITK/OpenCV did not accept serial execution settings.')
    return {
        'requested_threads': 1,
        'thread_control_limitations': limitations,
        'numpy_blas': blas,
        'suite_sha256': digest.hexdigest(),
        'itk_threads': sitk.ProcessObject.GetGlobalDefaultNumberOfThreads(),
        'opencv_threads': cv2.getNumThreads(),
        'scipy_fft_workers': fft.get_workers(),
        'native_pools': pools,
        'thread_environment': {key: os.environ.get(key) for key in THREAD_ENV},
        'zrad_version': zrad.__version__,
        'zrad_path': str(Path(zrad.__file__).resolve()),
        'dependencies': {dist.metadata['Name']: dist.version for dist in distributions()},
        'suite_version': 9,
    }
