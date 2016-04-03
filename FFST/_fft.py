"""
Try getting FFTs from pyFFTW, mklfft, numpy in that order of preference.
"""

from __future__ import division, absolute_import, print_function

import numpy as np
import warnings


try:
    import multiprocessing
    import pyfftw
    from functools import partial

    has_pyfftw = True

    pyfftw_threads = multiprocessing.cpu_count()
    pyfftw_planner_effort = 'FFTW_MEASURE'

    fft2 = partial(pyfftw.interfaces.numpy_fft.fft2,
                   planner_effort=pyfftw_planner_effort,
                   threads=pyfftw_threads)
    ifft2 = partial(pyfftw.interfaces.numpy_fft.ifft2,
                    planner_effort=pyfftw_planner_effort,
                    threads=pyfftw_threads)
    fft = partial(pyfftw.interfaces.numpy_fft.fft,
                  planner_effort=pyfftw_planner_effort,
                  threads=pyfftw_threads)
    ifft = partial(pyfftw.interfaces.numpy_fft.ifft,
                   planner_effort=pyfftw_planner_effort,
                   threads=pyfftw_threads)
    fftn = partial(pyfftw.interfaces.numpy_fft.fftn,
                   planner_effort=pyfftw_planner_effort,
                   threads=pyfftw_threads)
    ifftn = partial(pyfftw.interfaces.numpy_fft.ifftn,
                    planner_effort=pyfftw_planner_effort,
                    threads=pyfftw_threads)
    fftshift = pyfftw.interfaces.numpy_fft.fftshift
    ifftshift = pyfftw.interfaces.numpy_fft.ifftshift
    fftfreq = pyfftw.interfaces.numpy_fft.fftfreq

    # Turn on the cache for optimum performance
    pyfftw.interfaces.cache.enable()

    # increase cache preservation time from default of 0.1 seconds
    pyfftw.interfaces.cache.set_keepalive_time(5)

except ImportError as e:
    has_pyfftw = False
    try:
        warnings.warn("pyFFTW not found.  will try to use mklfft instead.")
        import mklfft
        fft = mklfft.fftpack.fft
        ifft = mklfft.fftpack.ifft
        fft2 = mklfft.fftpack.fft2
        ifft2 = mklfft.fftpack.ifft2
        fftn = mklfft.fftpack.fftn
        ifftn = mklfft.fftpack.ifftn
        fftshift = np.fft.fftshift
        ifftshift = np.fft.ifftshift
        fftfreq = np.fft.fftfreq
    except ImportError as e:
        warnings.warn("neither pyFFTW or mklfft found.  will use numpy.fft.")
        # Numpy's n-dimensional FFT routines may be using MKL, so prefered
        # over scipy
        fft = np.fft.fft
        ifft = np.fft.ifft
        fft2 = np.fft.fft2
        ifft2 = np.fft.ifft2
        fftn = np.fft.fftn
        ifftn = np.fft.ifftn
        fftshift = np.fft.fftshift
        ifftshift = np.fft.ifftshift
        fftfreq = np.fft.fftfreq


__all__ = ['fft', 'fft2', 'fftn', 'fftshift', 'fftfreq',
           'ifft', 'ifft2', 'ifftn', 'ifftshift',
           'fftnc', 'ifftnc', 'has_pyfftw']
if has_pyfftw:
    # the following functions are PyFFTW dependent
    __all__ += ['build_fftn', 'build_ifftn', 'pyfftw_threads']


# centered versions of fftn for convenience
def fftnc(a, s=None, axes=None, pre_shift_axes=None, post_shift_axes=None):
    y = ifftshift(a, axes=pre_shift_axes)
    y = fftn(y, s=s, axes=axes)
    return fftshift(y, axes=post_shift_axes)


def ifftnc(a, s=None, axes=None, pre_shift_axes=None, post_shift_axes=None):
    y = ifftshift(a, axes=pre_shift_axes)
    y = ifftn(y, s=s, axes=axes)
    return fftshift(y, axes=post_shift_axes)


if has_pyfftw:
    def build_fftn(a, fft_axes=None, threads=pyfftw_threads,
                   overwrite_input=False, planner_effort=pyfftw_planner_effort,
                   **kwargs):
        if not has_pyfftw:
            raise ValueError("pyfftw is required by plan_fftn")

        return pyfftw.builders.fftn(a,
                                    axes=fft_axes,
                                    threads=threads,
                                    overwrite_input=overwrite_input,
                                    planner_effort=planner_effort,
                                    **kwargs)

    def build_ifftn(a, fft_axes=None, threads=pyfftw_threads,
                    overwrite_input=False,
                    planner_effort=pyfftw_planner_effort, **kwargs):
        if not has_pyfftw:
            raise ValueError("pyfftw is required by plan_fftn")

        return pyfftw.builders.ifftn(a,
                                     axes=fft_axes,
                                     threads=threads,
                                     overwrite_input=overwrite_input,
                                     planner_effort=planner_effort,
                                     **kwargs)
