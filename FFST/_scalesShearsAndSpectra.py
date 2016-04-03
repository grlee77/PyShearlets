from __future__ import division, print_function, absolute_import

import numpy as np
import warnings
from .meyerShearlet import meyerShearletSpect, meyeraux


def _defaultNumberOfScales(l):
    numOfScales = int(np.floor(0.5 * np.log2(np.max(l))))
    if numOfScales < 1:
        raise ValueError('image to small!')
    return numOfScales


def scalesShearsAndSpectra(shape, numOfScales=None,
                           realCoefficients=True, maxScale='max',
                           shearletSpect=meyerShearletSpect,
                           shearletArg=meyeraux, realReal=True,
                           fftshift_spectra=True):
    """  Compute the shearlet spectra of a given shape and number of scales.

    The number of scales and a boolean indicating real or complex shearlets
    are optional parameters.

    Parameters
    ----------
    shape : array-like
        dimensions of the image
    numOfScales : int
        number of scales
    realCoefficients : bool
        Controls whether real or complex shearlets are generated.
    shearletSpect : string or handle
        shearlet spectrum
    shearletArg : ???
        further parameters for shearlet
    realReal : bool
        guarantee truly real shearlets
    maxScale : {'max', 'min'}, optional
        maximal or minimal finest scale

    Returns
    -------
    Psi : ndarray
        Shearlets in the Fourier domain.
    """
    if len(shape) != 2:
        raise ValueError("2D image dimensions required")

    if numOfScales is None:
        numOfScales = _defaultNumberOfScales(shape)

    # rectangular images
    if shape[1] != shape[0]:
        rectangular = True
    else:
        rectangular = False

    # for better symmetry each dimensions of the array should be odd
    shape = np.asarray(shape)
    shape_orig = shape.copy()
    shapem = np.mod(shape, 2) == 0  # True for even sized axes
    both_even = np.all(np.equal(shapem, False))
    both_odd = np.all(np.equal(shapem, True))
    shape[shapem] += 1

    if not realCoefficients:
        warnings.warn("Complex shearlet case may be buggy.  Doesn't "
                      "currently give perfect reconstruction.")

    if not (both_even or both_odd):
        # for some reason reconstruction is not exact in this case, so don't
        # allow it for now.
        raise ValueError("Mixture of odd and even array sizes is currently "
                         "unsupported.")

    # create meshgrid
    # largest value where psi_1 is equal to 1
    maxScale = maxScale.lower()
    if maxScale == 'max':
        X = 2**(2 * (numOfScales - 1) + 1)  # = 2^(2*numOfScales - 1)
    elif maxScale == 'min':
        X = 2**(2 * (numOfScales - 1))  # = 2^(2*numOfScales - 2)
    else:
        raise ValueError('Wrong option for maxScale, must be "max" or "min"')

    xi_x_init = np.linspace(0, X, (shape[1] + 1) / 2)
    xi_x_init = np.concatenate((-xi_x_init[-1:0:-1], xi_x_init), axis=0)
    if rectangular:
        xi_y_init = np.linspace(0, X, (shape[0] + 1) / 2)
        xi_y_init = np.concatenate((-xi_y_init[-1:0:-1], xi_y_init), axis=0)
    else:
        xi_y_init = xi_x_init

    # create grid, from left to right, bottom to top
    [xi_x, xi_y] = np.meshgrid(xi_x_init, xi_y_init[::-1], indexing='xy')

    # cones
    C_hor = np.abs(xi_x) >= np.abs(xi_y)  # with diag
    C_ver = np.abs(xi_x) < np.abs(xi_y)

    # number of shears: |-2^j,...,0,...,2^j| = 2 * 2^j + 1
    # now: inner shears for both cones:
    # |-(2^j-1),...,0,...,2^j-1|
    # = 2 * (2^j - 1) + 1
    # = 2^(j+1) - 2 + 1 = 2^(j+1) - 1
    # outer scales: 2 ("one" for each cone)
    # shears for each scale: hor: 2^(j+1) - 1, ver: 2^(j+1) - 1, diag: 2
    #  -> hor + ver + diag = 2*(2^(j+1) - 1) +2 = 2^(j + 2)
    #  + 1 for low-pass
    shearsPerScale = 2**(np.arange(numOfScales) + 2)
    numOfAllShears = 1 + shearsPerScale.sum()

    # init
    Psi = np.zeros(tuple(shape) + (numOfAllShears, ))
    # frequency domain:
    # k  2^j 0 -2^j
    #
    #     4  3  2  -2^j
    #      \ | /
    #   (5)- x -1  0
    #      / | \
    #              2^j
    #
    #        [0:-1:-2^j][-2^j:1:2^j][2^j:-1:1] (not 0)
    #           hor          ver        hor
    #
    # start with shear -2^j (insert in index 2^j+1 (with transposed
    # added)) then continue with increasing scale. Save to index 2^j+1 +- k,
    # if + k save transposed. If shear 0 is reached save -k starting from
    # the end (thus modulo). For + k just continue.
    #
    # then in time domain:
    #
    #  2  1  8
    #   \ | /
    #  3- x -7
    #   / | \
    #  4  5  6
    #

    # lowpass
    Psi[:, :, 0] = shearletSpect(xi_x, xi_y, np.NaN, np.NaN, realCoefficients,
                                 shearletArg, scaling_only=True)

    # loop for each scale
    for j in range(numOfScales):
        # starting index
        idx = 2**j
        start_index = 1 + shearsPerScale[:j].sum()
        shift = 1
        for k in range(-2**j, 2**j + 1):
            # shearlet spectrum
            P_hor = shearletSpect(xi_x, xi_y, 2**(-2 * j), k * 2**(-j),
                                  realCoefficients, shearletArg)
            if rectangular:
                P_ver = shearletSpect(xi_y, xi_x, 2**(-2 * j), k * 2**(-j),
                                      realCoefficients, shearletArg)
            else:
                # the matrix is supposed to be mirrored at the counter
                # diagonal
                # P_ver = fliplr(flipud(P_hor'))
                P_ver = np.rot90(P_hor, 2).T  # TODO: np.conj here too?
            if not realCoefficients:
                # workaround to cover left-upper part
                P_ver = np.rot90(P_ver, 2)

            if k == -2**j:
                Psi[:, :, start_index + idx] = P_hor * C_hor + P_ver * C_ver
            elif k == 2**j:
                Psi_idx = start_index + idx + shift
                Psi[:, :, Psi_idx] = P_hor * C_hor + P_ver * C_ver
            else:
                new_pos = np.mod(idx + 1 - shift, shearsPerScale[j]) - 1
                if(new_pos == -1):
                    new_pos = shearsPerScale[j] - 1
                Psi[:, :, start_index + new_pos] = P_hor
                Psi[:, :, start_index + idx + shift] = P_ver

                # update shift
                shift += 1

    # generate output with size shape_orig
    Psi = Psi[:shape_orig[0], :shape_orig[1], :]

    # modify spectra at finest scales to obtain really real shearlets
    # the modification has only to be done for dimensions with even length
    if realCoefficients and realReal and (shapem[0] or shapem[1]):
        idx_finest_scale = (1 + np.sum(shearsPerScale[:-1]))
        scale_idx = idx_finest_scale + np.concatenate(
            (np.arange(1, (idx_finest_scale + 1) / 2 + 1),
             np.arange((idx_finest_scale + 1) / 2 + 2, shearsPerScale[-1])),
            axis=0)
        scale_idx = scale_idx.astype(np.int)
        if shapem[0]:  # even number of rows -> modify first row:
            idx = slice(1, shape_orig[1])
            Psi[0, idx, scale_idx] = 1 / np.sqrt(2) * (
                Psi[0, idx, scale_idx] +
                Psi[0, shape_orig[1] - 1:0:-1, scale_idx])
        if shapem[1]:  # even number of columns -> modify first column:
            idx = slice(1, shape_orig[0])
            Psi[idx, 0, scale_idx] = 1 / np.sqrt(2) * (
                Psi[idx, 0, scale_idx] +
                Psi[shape_orig[0] - 1:0:-1, 0, scale_idx])

    if fftshift_spectra:
        # Note: changed to ifftshift so roundtrip tests pass for odd sized
        # arrays
        Psi = np.fft.ifftshift(Psi, axes=(0, 1))
    return Psi
