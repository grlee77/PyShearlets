from __future__ import division, print_function, absolute_import

import numpy as np

from .meyerShearlet import (meyerShearletSpect, meyeraux)

from ._scalesShearsAndSpectra import scalesShearsAndSpectra
from ._fft import fftshift, ifftshift, fftn, ifftn


def shearletTransformSpect(A, Psi=None, numOfScales=None,
                           realCoefficients=True, maxScale='max',
                           shearletSpect=meyerShearletSpect,
                           shearletArg=meyeraux, realReal=True):
    """
    #SHEARLETTRANSFORMSPECT compute shearlet transform
    # Compute the shearlet transform of a given image A. The number of scales
    # and a boolean indicating real or complex coefficients are optional
    # parameters.
    # Using a parameter value list further details can be provided.
    # ST contains the shearlet coefficients in a 3-d-matrix
    # where the third index indicates the respective shear (1 for lowpass,
    # 2:end for the different shears and scales). The images are ordered
    # ascending with the scale and within each scale counter-clockwise with the
    # direction of the shear.
    # Psi contains the respective shearlets (in Fourier domain).
    #
    # INPUT:
    #  A                (matrix) image (or data) to transform
    #  numOfScales      (int) number of scales OR
    #  realCoefficients (bool) real/complex shearlets  (optional)
    #
    # OUTPUT:
    #  ST               (3-d-matrix) shearlet coefficients
    #  Psi              (3-d-matrix) spectrum of shearlets
    #
    # PARAMETERS: (as optional parameter value list, arbitrary order)
    #  'shearletSpect'  (string or def handle) shearlet spectrum
    #  'shearletArg'    (arbitrary) further parameters for shearlet
    #  'realReal'       (bool) guarantees really real shearlets
    #  'maxScale'       ('max','min') maximal or minimal finest scale
    #
    # EXAMPLES:
    #  #compute shearlet transform of image A with default parameters
    #  [ST,Psi] = shearletTransformSpect(A)
    #
    #  #compute shearlet transform of image A with precomputed shearlet spectrum
    #  [ST,Psi] = shearletTransformSpect(A,Psi)
    #
    #  #compute sharlet transform of image A with specified number of scales
    #  [ST,Psi] = shearletTransformSpect(A, numOfScales=4)
    #
    #  #compute shearlet transform of image A with own shearlet
    #  [ST,Psi] = shearletTransformSpect(A, shearletSpect=yourShearletSpect)
    #
    #--------------------------------------------------------------------------
    # Sören Häuser ~ FFST ~ 2014-07-22 ~ last edited: 2014-07-22 (Sören Häuser)
    """
    # parse input
    A = np.asarray(A)
    if (A.ndim != 2) or np.any(np.asarray(A.shape) <= 1):
        raise ValueError("2D image required")

    # compute spectra
    if Psi is None:
        l = A.shape
        if numOfScales is None:
            numOfScales = int(np.floor(0.5 * np.log2(np.max(l))))
            if numOfScales < 1:
                raise ValueError('image to small!')
        Psi = scalesShearsAndSpectra(l, numOfScales=numOfScales,
                                     realCoefficients=realCoefficients,
                                     shearletSpect=meyerShearletSpect,
                                     shearletArg=meyeraux)

    # shearlet transform
    if False:
        # INCORRECT TO HAVE FFTSHIFT SINCE Psi ISNT SHIFTED!
        uST = Psi * fftshift(fftn(A))[..., np.newaxis]
        ST = ifftn(ifftshift(uST, axes=(0, 1)), axes=(0, 1))
    else:
        uST = Psi * fftn(A)[..., np.newaxis]
        ST = ifftn(uST, axes=(0, 1))

    # due to round-off errors the imaginary part is not zero but very small
    # -> neglect it
    if realCoefficients and realReal and np.isrealobj(A):
        ST = ST.real

    return (ST, Psi)
