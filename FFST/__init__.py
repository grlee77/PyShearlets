from __future__ import absolute_import

from ._scalesShearsAndSpectra import scalesShearsAndSpectra
from ._shearletTransformSpect import shearletTransformSpect
from ._inverseShearletTransformSpect import inverseShearletTransformSpect

from numpy.testing import Tester
test = Tester().test
bench = Tester().bench

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
