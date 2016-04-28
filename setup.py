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

import subprocess


MAJOR = 0
MINOR = 5
MICRO = 0
ISRELEASED = True
VERSION = '%d.%d.%d' % (MAJOR, MINOR, MICRO)


# Return the git revision as a string
def git_version():
    def _minimal_ext_cmd(cmd):
        # construct minimal environment
        env = {}
        for k in ['SYSTEMROOT', 'PATH']:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        # LANGUAGE is used on win32
        env['LANGUAGE'] = 'C'
        env['LANG'] = 'C'
        env['LC_ALL'] = 'C'
        out = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                               env=env).communicate()[0]
        return out

    try:
        out = _minimal_ext_cmd(['git', 'rev-parse', 'HEAD'])
        GIT_REVISION = out.strip().decode('ascii')
    except OSError:
        GIT_REVISION = "Unknown"

    return GIT_REVISION


def get_version_info():
    # Adding the git rev number needs to be done inside
    # write_version_py(), otherwise the import of FFST.version messes
    # up the build under Python 3.
    FULLVERSION = VERSION
    if os.path.exists('.git'):
        GIT_REVISION = git_version()
    elif os.path.exists('FFST/version.py'):
        # must be a source distribution, use existing version file
        # load it as a separate module to not load FFST/__init__.py
        import imp
        version = imp.load_source('FFST.version', 'FFST/version.py')
        GIT_REVISION = version.git_revision
    else:
        GIT_REVISION = "Unknown"

    if not ISRELEASED:
        FULLVERSION += '.dev0+' + GIT_REVISION[:7]

    return FULLVERSION, GIT_REVISION


def write_version_py(filename='FFST/version.py'):
    cnt = """
# THIS FILE IS GENERATED FROM PYSHEARLETS SETUP.PY
short_version = '%(version)s'
version = '%(version)s'
full_version = '%(full_version)s'
git_revision = '%(git_revision)s'
release = %(isrelease)s

if not release:
    version = full_version
"""
    FULLVERSION, GIT_REVISION = get_version_info()

    a = open(filename, 'w')
    try:
        a.write(cnt % {'version': VERSION,
                       'full_version': FULLVERSION,
                       'git_revision': GIT_REVISION,
                       'isrelease': str(ISRELEASED)})
    finally:
        a.close()


def configuration(parent_package='', top_path=None):
    if os.path.exists('MANIFEST'):
        os.remove('MANIFEST')

    config = Configuration(None, parent_package, top_path)

    # main modules
    config.add_subpackage('FFST')

    return config

if __name__ == '__main__':
    write_version_py()
    setup(name='PyShearlets',
          version='1.0',
          description='fast finite shearlet transform',
          author='Gregory R. Lee',
          author_email='grlee77@gmail.com',
          url='https://bitbucket.org/grlee77/FFST',
          configuration=configuration)
