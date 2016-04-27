from __future__ import division, print_function, absolute_import

import numpy as np


def _jk2index(j, k, cone):
    """helper function, compute index from j, k and cone. """
    # lowpass
    index = 1
    if np.isnan(j) and np.isnan(k) and cone == '0':
        return None
    else:
        # sum of lower scales
        index = index + np.sum(2**(2+np.arange(j)))

        # get detail index from shear (and cone!)
        if cone == 'h':
            if k <= 0:
                index = index - k
            else:
                index = index + 4*2**j - k
        elif cone == 'v':
            index = index + 2**j + (k + 2**j)
        elif cone == 'x':
            # TODO: if k can be complex, will need to fix this
            index = index + (2 + np.sign(k)) * 2**j

        # sligth adjustment ( k=0 <=> index = 1)
        index += 1

    return index


def _index2jk(index):
    """helper function: compute j, k and cone from index. """
    if index <= 1:  # lowpass, j and k not needed:
        j = np.NaN
        k = np.NaN
        cone = '0'
    else:
        # substract 1 for the lowpass
        index = index - 1

        # determine scale j
        # substract number of shears in each scale:
        # 2**(j+0), 2**(j+1), 2**(j+2),
        j = 0
        while index > 2**(2 + j):
            index = index - 2**(j+2)
            j = j + 1

        # shift to zero (first index <=> k=0)
        index = index - 1

        # determine cone
        # index | 0 1 ... 2**j ... 2*2**j ... 3*2**j ... 4*2**j -1
        # k     | 0 -1   -2**j       0        2**j        1
        # cone  | h ... h  x  v ... v ... v   x h ...   h
        index2 = index / 2**j

        if index2 < 1:
            k = -index
            cone = 'h'
        elif index2 == 1:
            k = -2**j
            cone = 'x'
        elif index2 < 3:
            k = index - 2*2**j
            cone = 'v'
        elif index2 == 3:
            k = 2**j
            cone = 'x'
        else:
            k = -(index - 4*2**j)
            cone = 'h'

    return (j, k, cone)


def shearletScaleShear(a, b=None, c=None, d=None):
    """ compute index from scale j, shear k and cone and vice versa.

    Optionally return values of shearlets (or coefficients) for given index or
    given scale j, shear k and cone.

    # TODO: convert the Matlab-formated docstring below

    #
    # OPTIONS
    ## scale, shear and cone from index
    # [j,k,cone] = shearletScaleShear(a)
    # INPUT:
    #  a    (int) index
    #
    # OUTPUT:
    #  j        (int) scale j (>= 0)
    #  k        (int) shear k, -2**j <= k <= 2**j
    #  cone     (char) cone [h,v,x,0]
    ## return data for index
    # ST = shearletScaleShear(a,b)
    # INPUT:
    #  a    (3-d-matrix) shearlets or shearlet coefficients
    #  b    (int) index
    #
    # OUTPUT:
    #  ST       (matrix) layer of ST for index [ST[:, :,index)]
    ## index from scale, shear and cone
    # index = shearletScaleShear(a,b,c)
    # INPUT:
    #  a        (int) scale j (>= 0)
    #  b        (int) shear k, -2**j <= k <= 2**j
    #  c        (char) cone [h,v,x,0]
    #
    # OUTPUT:
    #  index    (int) respective index
    ## return data for j,k and cone
    # ST = shearletScaleShear(a,b,c,d)
    # INPUT:
    #  a        (3-d-matrix) shearlets or shearlet coefficients
    #  b        (int) scale j (>= 0)
    #  c        (int) shear k, -2**j <= k <= 2**j
    #  d        (char) cone [h,v,x,0]
    #
    # OUTPUT:
    #  index    (int) respective index
    #

    """
    # display informations
    disp = False

    # different cases
    if b is None:
        # compute j and k from index
        index = a
        (j, k, cone) = _index2jk(index)
        varargout = (j, k, cone)
        if disp:
            print('index %d represents:\n' % index)
            print('scale j: %d (a = %.4f)\n' % (j, 4**(-j)))
            print('shear k: %d (s = %.4f)\n' % (k, 2**(-j)*k))
            print('cone   : %s\n', cone)
    elif c is None:
        # return data for index
        ST = a
        index = b
        varargout = ST[:, :, index]
        (j, k, cone) = _index2jk(index)

        if disp:
            print('index %d represents:\n' % index)
            print('scale j: %d (a = %.4f)\n' % (j, 4**(-j)))
            print('shear k: %d (s = %.4f)\n' % (k, 2**(-j)*k))
            print('cone   : %s\n', cone)
    elif d is None:
            # compute index from j and k and cone
            j = a
            k = b
            cone = c
            index = _jk2index(j, k, cone)
            varargout = [index]

            if disp:
                print('index %d represents:\n' % index)
                print('scale j: %d (a = %.4f)\n' % (j, 4**(-j)))
                print('shear k: %d (s = %.4f)\n' % (k, 2**(-j)*k))
                print('cone   : %s\n', cone)
    else:
        # return data for j and k and cone
        ST = a
        j = b
        k = c
        cone = d

        index = _jk2index(j, k, cone)
        varargout = ST[:, :, index]

        if disp:
            print('index %d represents:\n' % index)
            print('scale j: %d (a = %.4f)\n' % (j, 4**(-j)))
            print('shear k: %d (s = %.4f)\n' % (k, 2**(-j)*k))
            print('cone   : %s\n', cone)

    return varargout
