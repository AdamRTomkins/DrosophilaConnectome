#!/usr/bin/env python

import sys, os
from glob import glob

# Install setuptools if it isn't available:
try:
    import setuptools
except ImportError:
    from ez_setup import use_setuptools
    use_setuptools()

from distutils.command.install_headers import install_headers
from setuptools import find_packages
from setuptools import setup

NAME =               'processor_component'
VERSION =            '0.2'
AUTHOR =             'Adam Tomkins'
AUTHOR_EMAIL =       'a.tomkins@shef.ac.uk'
MAINTAINER =         AUTHOR
MAINTAINER_EMAIL =   AUTHOR_EMAIL
DESCRIPTION =        'A wrapper for the OSP processor component'
URL =                'TBD'
LONG_DESCRIPTION =   DESCRIPTION
DOWNLOAD_URL =       URL
LICENSE =            'BSD'
CLASSIFIERS = [
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: BSD License',
    'Operating System :: OS Independent',
    'Programming Language :: Python',
    'Topic :: Scientific/Engineering',
    'Topic :: Software Development']

# Explicitly switch to parent directory of setup.py in case it
# is run from elsewhere:
os.chdir(os.path.dirname(os.path.realpath(__file__)))
PACKAGES =           find_packages()

if __name__ == "__main__":
    if os.path.exists('MANIFEST'):
        os.remove('MANIFEST')

    setup(
        name = NAME,
        version = VERSION,
        author = AUTHOR,
        author_email = AUTHOR_EMAIL,
        license = LICENSE,
        classifiers = CLASSIFIERS,
        description = DESCRIPTION,
        long_description = LONG_DESCRIPTION,
        url = URL,
        maintainer = MAINTAINER,
        maintainer_email = MAINTAINER_EMAIL,
        packages = PACKAGES,
        include_package_data = True,
        install_requires = [
        ],
        eager_resources = ['processor_component/data/auth_dict.json', 'processor_component/data/email_dict.json']
        )
