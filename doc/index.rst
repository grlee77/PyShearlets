PyShearlets - Fast Finite Shearlet Transforms in Python
=======================================================

The PyShearlets package provides a fast implementation of the Finite Shearlet
Transform (FFST).  Following the path via the continuous shearlet
transform, its counterpart on cones and finally its discretization on the full
grid we obtain the translation invariant discrete shearlet transform. This
discrete shearlet transform can be efficiently computed by the fast Fourier
transform (FFT).  The discrete shearlets constitute a Parseval frame of the
finite Euclidean space.

More information can be found in the following Papers

   - S. Häuser and G. Steidl, "Convex Multilabel Segmentation with Shearlet
     Regularization",  International Journal of Computer Mathematics.
     90, (1), 62-81, 2013

   - S. Häuser and G. Steidl, "FFST: a tutorial",
     arXiv Preprint 1202.1773, 2014

The Matlab-Version of the toolbox is available for free download at

http://www.mathematik.uni-kl.de/imagepro/software/

Requirements
------------

PyShearlets is a package for the Python programming language. It requires:

 - Python_ 2.7 or >=3.3
 - Numpy_ >= 1.7

 Optional Dependencies
 ---------------------
 The following packages provide a faster implementation of the FFT.
 - PyFFTW_ :  if this package is found, it will be used in place of numpy.fft
 - mklfft_:  if PyFFTW is not found, this package will be used for FFTs

Download
--------

The most recent *development* version can be found on GitHub at
https://github.com/PyShearlets/pywt.

Latest release, including source and binary package for Windows, is available
for download from the `Python Package Index`_ or on the `Releases Page`_.

Install
-------

 - Install PyShearlets with ``pip install PyShearlets``.

 - To build and install from source, navigate to the downloaded PyShearlets
   source code directory and type ``python setup.py install``.

Documentation
-------------

For a couple of basic usage examples see the `examples` directory in the
source package.

Contact
-------

Use `GitHub Issues`_ to post comments or questions.

License
-------

PyShearlets is a released under a GPL v3 license.

.. _GitHub: https://github.com/grlee77/PyShearlets
.. _GitHub Issues: https://github.com/grlee77/PyShearlets/issues
.. _Numpy: http://www.numpy.org
.. _mklfft: https://docs.continuum.io/accelerate/mkl_fft
.. _Python: http://python.org/
.. _PyFFTW: https://github.com/pyFFTW/pyFFTW
.. _Python Package Index: http://pypi.python.org/pypi/PyShearlets/
.. _Releases Page: https://github.com/PyShearlets/pywt/releases
