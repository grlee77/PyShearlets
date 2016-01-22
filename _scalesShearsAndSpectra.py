from __future__ import division, print_function, absolute_import

import numpy as np
from .meyerShearlet import meyerShearletSpect, meyeraux


def _defaultNumberOfScales(l):
    numOfScales = int(np.floor(0.5 * np.log2(np.max(l))))
    if numOfScales < 1:
        raise ValueError('image to small!')
    return numOfScales


def scalesShearsAndSpectra(l, numOfScales=None,
                           realCoefficients=True, maxScale='max',
                           shearletSpect=meyerShearletSpect,
                           shearletArg=meyeraux, realReal=True):
    """
    #SCALESSHEARSANDSPECTRA compute shearlet spectra
    # Compute the shearlet spectra of a given size l. The number of scales
    # and a boolean indicating real or complex shearlets are optional
    # parameters.
    # Using a parameter value list further details can be provided.
    # The output Psi is a 3-d-matrix with the shearlets in Fourier domain
    # ordered with ascending scale and within each scale ordered by the
    # direction of the respective shears (see comments below for further
    # details).
    #
    # INPUT:
    #  l                (vector) dimensions of image
    #  numOfScales      (int) number of scales OR
    #                   (3-d-matrix) precomputed Psi (optional)
    #                   Note that for internal use the first element of
    #                   varargin can also be a struct passed from
    #                   shearletTransformSpect.m.
    #  realCoefficients (bool) real/complex shearlets  (optional)
    #
    # OUTPUT:
    #  Psi              (3-d-matrix) spectrum of shearlets
    #
    # PARAMETERS: (as optional parameter value list, arbitrary order)
    #  'shearletSpect'  (string or def handle) shearlet spectrum
    #  'shearletArg'    (arbitrary) further parameters for shearlet
    #  'realReal'       (bool) guarantees really real shearlets
    #  'maxScale'       ('max','min') maximal or minimal finest scale
    #
    # EXAMPLES:
    #  see shearletTransformSpect.m
    #
    #--------------------------------------------------------------------------
    # Sören Häuser ~ FFST ~ 2014-07-22 ~ last edited: 2014-07-22 (Sören Häuser)
    """
    ## parse input

    if len(l) != 2:
        raise ValueError("2D image dimensions required")

    if numOfScales is None:
        numOfScales = _defaultNumberOfScales(l)

    ## rectangular images
    if l[1] != l[0]:
        rectangular = True
    else:
        rectangular = False

    ## computation of the shearlets

    #for better symmetry each l should be odd
    l = np.asarray(l)
    l_orig = l.copy()
    lm = (1 - np.mod(l, 2)) > 0  # True for even, False for odd
    l[lm] += 1

    # create meshgrid
    #largest value where psi_1 is equal to 1
    if maxScale == 'max':
        X = 2**(2*(numOfScales-1)+1)  # = 2^(2*numOfScales - 1)
    elif maxScale == 'min':
        X = 2**(2*(numOfScales-1))  # = 2^(2*numOfScales - 2)
    else:
        raise ValueError('Wrong option for maxScale, must be "max" or "min"')

    #xi_x = linspace(-X,X-1/l(2)*2*X,l(2)); #not exactly symmetric
    xi_x_init = np.linspace(0, X, (l[1]+1)/2)
    xi_x_init = np.concatenate((-xi_x_init[-1:0:-1],
                                xi_x_init), axis=0)
    if rectangular:
        xi_y_init = np.linspace(0, X, (l[0]+1)/2)
        xi_y_init = np.concatenate((-xi_y_init[-1:0:-1],
                                    xi_y_init), axis=0)
    else:
        xi_y_init = xi_x_init

    #create grid, from left to right, bottom to top
    [xi_x, xi_y] = np.meshgrid(xi_x_init, xi_y_init[::-1], indexing='xy')

    #cones
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

    #init
    Psi = np.zeros(tuple(l) + (numOfAllShears, ))
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

    #lowpass
    Psi[:, :, 0] = shearletSpect(xi_x, xi_y, np.NaN, np.NaN, realCoefficients,
                                 shearletArg, scaling_only=True)

    #loop for each scale
    for j in range(numOfScales):
        #starting index
        idx = 2**j
        start_index = 1 + shearsPerScale[:j].sum()
        shift = 1
        for k in range(-2**j, 2**j + 1):
            #shearlet spectrum
            P_hor = shearletSpect(xi_x, xi_y, 2**(-2*j), k*2**(-j),
                                  realCoefficients, shearletArg)
            if rectangular:
                P_ver = shearletSpect(xi_y, xi_x, 2**(-2*j), k*2**(-j),
                                      realCoefficients, shearletArg)
            else:
                #the following three terms are equivalent
                #the matrix is supposed to be mirrored at the counter
                #diagonal
                #P_ver = fliplr(flipud(P_hor'))
                #P_ver = rot90(rot90(P_hor)')
                P_ver = np.rot90(P_hor, 2).T  # np.conj too?
            if not realCoefficients:
                # workaround to cover left-upper part
                P_ver = np.rot90(P_ver, 2)
            if k == -2**j:
                if realCoefficients:
                    Psi[:, :, start_index + idx] = \
                        P_hor * C_hor + P_ver * C_ver
                else:
                    Psi[:, :, start_index + idx] = \
                        P_hor * C_hor + P_ver * C_ver
            elif k == 2**j:
                Psi[:, :, start_index + idx + shift] = \
                    P_hor * C_hor + P_ver * C_ver
            else:

                new_pos = np.mod(idx+1-shift, shearsPerScale[j])-1
                if(new_pos == -1):
                    new_pos = shearsPerScale[j] - 1
                Psi[:, :, start_index + new_pos] = P_hor
                Psi[:, :, start_index + idx + shift] = P_ver

                #update shift
                shift += 1

    #generate output with size l
    Psi = Psi[:l_orig[0], :l_orig[1], :]

    #modify spectra at finest scales to obtain really real shearlets
    #the modification has only to be done for dimensions with even length
    if realCoefficients and realReal and (lm[0] or lm[1]):
        idx_finest_scale = (1 + np.sum(shearsPerScale[:-1]))
        scale_idx = idx_finest_scale + np.concatenate(
            (np.arange(1, (idx_finest_scale+1)/2+1),
             np.arange((idx_finest_scale+1)/2+2, shearsPerScale[-1])), axis=0)
        scale_idx = scale_idx.astype(np.int)
        if lm[0]:  # even number of rows -> modify first row:
            idx = slice(1, l_orig[1])
            Psi[0, idx, scale_idx] = 1/np.sqrt(2)*Psi[0, idx, scale_idx] + \
                1/np.sqrt(2)*Psi[0, l_orig[1]-1:0:-1, scale_idx]
        if lm[1]:  # even number of columns -> modify first column:
            idx = slice(1, l_orig[0])
            Psi[idx, 0, scale_idx] = 1/np.sqrt(2)*Psi[idx, 0, scale_idx] + \
                1/np.sqrt(2)*Psi[l_orig[0]-1:0:-1, 0, scale_idx]
    return Psi
