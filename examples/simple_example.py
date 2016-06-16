from __future__ import division, print_function, absolute_import

import numpy as np
from numpy.testing import assert_
from FFST import (shearletTransformSpect, scalesShearsAndSpectra,
                  inverseShearletTransformSpect)
from pyir.utils import ImageGeometry, ellipse_im

# simple example for FFST
# computes the shearlet transform of some geometric image
#
#--------------------------------------------------------------------------
# 2012-01-20, v1.0, (c) Sören Häuser
ig = ImageGeometry(nx=128, ny=128, dx=1)
phantom, params = ellipse_im(ig, params='shepplogan-mod')  #

compare_to_matlab = False
if True:
    if compare_to_matlab:
        matlab_data = scipy.io.loadmat('/media/Data1/src_repositories/my_git/pyrecon/misc_codes_of_possible_use/Wavelet - Shearlets/FFST - Fast Finite Shearlet Transform/FFST_v2/tmp_matlab.mat')
        Psi_Matlab = matlab_data['Psi']
        A_Matlab = matlab_data['A']
        ST_Matlab = matlab_data['ST']
        A = A_Matlab.copy()
    else:
        A = phantom[:, ::-1].T.copy()
    # numOfScales=None
    # realCoefficients=True
    # maxScale='max'
    # shearletSpect=meyerShearletSpect,
    # shearletArg=meyeraux
    # realReal=True
precompute = True
if not precompute:
    # shearlet transform
    [ST, Psi] = shearletTransformSpect(A)
else:
    # precompute Psi
    Psi = scalesShearsAndSpectra(A.shape, numOfScales=None,
                                 realCoefficients=True)
    # apply transform using precomputed Psi
    ST, Psi = shearletTransformSpect(A, Psi)

# inverse shearlet transform
C = inverseShearletTransformSpect(ST, Psi)

assert_((A-C).max() < 1e-13)

if compare_to_matlab:
    assert_((ST-ST_Matlab).max() < 1e-13)
    assert_((A-A_Matlab).max() < 1e-13)
    assert_((Psi-Psi_Matlab).max() < 1e-13)

volshow(Psi)
volshow(ST)
volshow([A, C])
