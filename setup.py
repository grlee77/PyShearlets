#!/usr/bin/env python
from setuptools import setup, find_packages

import versioneer

setup(name='PyShearlets',
      packages=find_packages(),
      version=versioneer.get_version(),
      cmdclass=versioneer.get_cmdclass(),
      description='2D fast finite shearlet transforms.',
      author='Gregory R. Lee',
      author_email='grlee77@gmail.com',
      url='https://bitbucket.org/grlee77/FFST',
      license='BSD 3-clause',
      zip_safe=False,
      # ackage_data={},
      )
