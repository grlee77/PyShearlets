#!/usr/bin/env python
'''
Installation script for FFST

Note:
To make a source distribution:
python setup.py sdist

To make an RPM distribution:
python setup.py bdist_rpm

To Install:
python setup.py install --prefix=/usr/local

See also:
python setup.py bdist --help-formats
'''

import os
from numpy.distutils.core import setup
from numpy.distutils.misc_util import Configuration
import versioneer


def configuration(parent_package='', top_path=None):
    if os.path.exists('MANIFEST'):
        os.remove('MANIFEST')

    config = Configuration(None, parent_package, top_path)

    # main modules
    config.add_subpackage('FFST')
    config.add_subpackage('FFST.tests')

    return config


setup(name='PyShearlets',
      version=versioneer.get_version(),
      cmdclass=versioneer.get_cmdclass(),
      description='2D fast finite shearlet transforms.',
      author='Gregory R. Lee',
      author_email='grlee77@gmail.com',
      url='https://bitbucket.org/grlee77/FFST',
      configuration=configuration)
