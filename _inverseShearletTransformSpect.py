from __future__ import division, print_function, absolute_import

import numpy as np

from .meyerShearlet import (meyerShearletSpect, meyeraux)
from ._scalesShearsAndSpectra import scalesShearsAndSpectra
from grl_utils import fftshift, ifftshift, fftn, ifftn


def inverseShearletTransformSpect(ST, Psi=None, maxScale='max',
                                  shearletSpect=meyerShearletSpect,
                                  shearletArg=meyeraux):
    """
    #INVERSESHEARLETTRANSFORMSPECT compute inverse shearlet transform
    # Compute the inverse shearlet transform for given shearlet coefficients ST.
    # If the shearlet spectra are not given they are computed using parameters
    # guessed from the coefficients.
    # The parameters 'shearletSpect', 'shearletArg' and 'maxScale' cannot be
    # guessed and have to be provided if not the default ones.
    #
    # INPUT:
    #  ST               (3-d-matrix) shearlet transform
    #  Psi              (3-d-matrix) spectrum of shearlets (optional)
    #
    # OUTPUT:
    #  A                (matrix) reconstructed image
    #
    # PARAMETERS: (as optional parameter value list, arbitrary order)
    #  'shearletSpect'  (string or def handle) shearlet spectrum
    #  'shearletArg'    (arbitrary) further parameters for shearlet
    #  'maxScale'       ('max','min') maximal or minimal finest scale
    #
    #--------------------------------------------------------------------------
    # Sören Häuser ~ FFST ~ 2014-07-22 ~ last edited: 2014-07-22 (Sören Häuser)
    """

    if Psi is None:
        #numOfScales
        #possible: 1, 4, 8, 16, 32,
        # -> -1 for lowpass
        # -> divide by for (1, 2, 4, 8,
        # -> +1 results in a 2^# number -> log returns #
        numOfScales = int(np.log2((ST.shape[-1] - 1)/4 + 1))

        #realCoefficients
        realCoefficients = True  #np.isrealobj(ST[:, :, 1])

        #realReal
        realReal = True  #np.isrealobj(ST[:, :, -1])

        #compute spectra
        Psi = scalesShearsAndSpectra((ST.shape[0], ST.shape[1]),
                                     numOfScales=numOfScales,
                                     realCoefficients=realCoefficients,
                                     realReal=realReal)

    ## inverse shearlet transform
    A = fftshift(fftn(ST, axes=(0, 1)), axes=(0, 1)) * Psi
    A = A.sum(axis=-1)
    A = ifftn(ifftshift(A)).real
    if np.isrealobj(ST):
        A = A.real

    return A
