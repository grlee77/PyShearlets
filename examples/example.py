import numpy as np
from matplotlib import pyplot as plt

import skimage.data
from skimage import img_as_float
from skimage.transform import resize

from FFST import (scalesShearsAndSpectra,
                  inverseShearletTransformSpect,
                  shearletTransformSpect)


def ifft2(x):
    """ centered inverse FFT """
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(Psi[..., idx])))

X = img_as_float(skimage.data.camera())
X = resize(X, (256, 256))

# compute shearlet transform
ST, Psi = shearletTransformSpect(X)

Psi = scalesShearsAndSpectra(X.shape, numOfScales=None,
                             realCoefficients=True)

idx = 13
fig, axes = plt.subplots(2, 2)

axes[0, 0].imshow(X, interpolation='nearest', cmap=plt.cm.gray)
axes[0, 0].set_axis_off()
axes[0, 0].set_title('original image')

axes[0, 1].imshow(ST[..., idx], interpolation='nearest', cmap=plt.cm.gray)
axes[0, 1].set_axis_off()
axes[0, 1].set_title('shearlet coefficients')

Psi_shifted = np.fft.fftshift(Psi[..., idx])
axes[1, 0].imshow(Psi_shifted, interpolation='nearest', cmap=plt.cm.gray)
axes[1, 0].set_axis_off()
axes[1, 0].set_title('shearlet in Fourier domain')

axes[1, 1].imshow(np.abs(ifft2(Psi_shifted)),
                  interpolation='nearest', cmap=plt.cm.gray)
axes[1, 1].set_axis_off()
axes[1, 1].set_title('shearlet in time domain')

# show frame tightness and exactness

plt.figure()
plt.imshow(1 - np.sum(Psi**2, -1), cmap=plt.cm.gray)
plt.colorbar()
plt.title('Frame Tightness')

XX = inverseShearletTransformSpect(ST, Psi)

plt.figure()
plt.imshow(np.abs(X-XX), cmap=plt.cm.gray)
plt.colorbar()
plt.title('Transform Exactness')


# compute shearlet transform
ST, Psi = shearletTransformSpect(X, realCoefficients=False)
idx = 13
fig, axes = plt.subplots(3, 2)

from mpl_toolkits.axes_grid1 import make_axes_locatable
def add_cbar(im, ax):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="10%", pad=0.05)
    plt.colorbar(im, cax=cax)

cmap = plt.cm.jet
im = axes[0, 0].imshow(X, interpolation='nearest', cmap=cmap)
add_cbar(im0, axes[0, 0])
axes[0, 0].set_axis_off()
axes[0, 0].set_title('original image')
axes[0, 1].set_axis_off()

im = axes[1, 0].imshow(ST[..., idx].real, interpolation='nearest', cmap=cmap)
add_cbar(im, axes[1, 0])
axes[1, 0].set_axis_off()
axes[1, 0].set_title('shearlet coefficients (real part)')

im = axes[1, 1].imshow(ST[..., idx].imag, interpolation='nearest', cmap=cmap)
add_cbar(im, axes[1, 1])
axes[1, 1].set_axis_off()
axes[1, 1].set_title('shearlet coefficients (imaginary part)')

im = axes[2, 0].imshow(np.abs(ST[..., idx]), interpolation='nearest', cmap=cmap)
add_cbar(im, axes[2, 0])
axes[2, 0].set_axis_off()
axes[2, 0].set_title('shearlet coefficients (absolute value)')

im = axes[2, 1].imshow(np.angle(ST[..., idx]), interpolation='nearest', cmap=cmap)
add_cbar(im, axes[2, 1])
axes[2, 1].set_axis_off()
axes[2, 1].set_title('shearlet coefficients (phase)')
