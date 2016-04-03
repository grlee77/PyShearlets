import numpy as np

from numpy.testing import assert_, assert_raises, assert_equal

from FFST import (scalesShearsAndSpectra,
                  inverseShearletTransformSpect,
                  shearletTransformSpect)


def test_tight_frame():
    for shape in [(32, 32), (64, 64), (128, 128), (256, 256)]:
        Psi = scalesShearsAndSpectra(shape)
        assert_equal(Psi[0, 0, 0], 1)  # If FFTshifted will be 1 at corner
        # sum along last axis = 1 everywhere if it is a tight frame
        assert_(np.max(1 - np.sum(Psi**2, -1)) < 1e-14)

    # test odd shape
    for shape in [(65, 65), (125, 125)]:
        Psi = scalesShearsAndSpectra(shape)
        assert_equal(Psi[0, 0, 0], 1)
        assert_(np.max(1 - np.sum(Psi**2, -1)) < 1e-14)

    # test mixture of odd and even
    for shape in [(64, 65), (65, 64)]:
        # Psi = scalesShearsAndSpectra(shape, realReal=False)
        # assert_(np.max(1 - np.sum(Psi**2, -1)) < 1e-14)
        assert_raises(ValueError, scalesShearsAndSpectra, shape)

    for shape in [(32, 32), (33, 33)]:
        for maxScale in ['max', 'min']:
            for realReal in [True, False]:
                Psi = scalesShearsAndSpectra(shape, maxScale=maxScale,
                                             realReal=realReal)
                # sum along last axis = 1 everywhere if it is a tight frame
                assert_(np.max(1 - np.sum(Psi**2, -1)) < 1e-14)


def test_perfect_recon():
    rstate = np.random.RandomState(1234)
    for shape in [(32, 32), (64, 64), (128, 128)]:
        X = rstate.standard_normal(shape)
        ST, Psi = shearletTransformSpect(X, realCoefficients=True)
        XX = inverseShearletTransformSpect(ST, Psi)
        # sum along last axis = 1 everywhere if it is a tight frame
        assert_(np.max(X - XX) < 1e-13)

    # test odd shape
    for shape in [(65, 65), (125, 125)]:
        X = rstate.standard_normal(shape)
        ST, Psi = shearletTransformSpect(X, realCoefficients=True)
        XX = inverseShearletTransformSpect(ST, Psi)
        # sum along last axis = 1 everywhere if it is a tight frame
        assert_(np.max(X - XX) < 1e-13)

    # check some other non-default settings
    for shape in [(32, 32), (33, 33)]:
        for maxScale in ['max', 'min']:
            for realReal in [True, False]:
                X = rstate.standard_normal(shape)
                ST, Psi = shearletTransformSpect(X,
                                                 realCoefficients=True,
                                                 maxScale=maxScale,
                                                 realReal=realReal)
                XX = inverseShearletTransformSpect(ST, Psi)
                # sum along last axis = 1 everywhere if it is a tight frame
                assert_(np.max(X - XX) < 1e-13)

    # # test mixture of odd and even
    # for shape in [(64, 65), (65, 64)]:
    #     X = rstate.standard_normal(shape)
    #     ST, Psi = shearletTransformSpect(X, realCoefficients=True,
    #                                      realReal=False)
    #     XX = inverseShearletTransformSpect(ST, Psi)
    #     # sum along last axis = 1 everywhere if it is a tight frame
    #     assert_(np.max(X - XX) < 1e-2)  # TODO: get close if realReal=False, but must be a small bug somewhere in mixed odd/even case
