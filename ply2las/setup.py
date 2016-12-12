#!/usr/bin/env python

from setuptools import setup, find_packages

from codecs import open
from os import path

here=path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description=f.read()

print find_packages()

setup(
        name='extractor_ply2las',
        version='2.0.0',

        description='convert 3D PLY files to LAS format',
        long_description=long_description,

        entry_points={
            'console_scripts': [
                'terra.ply2las.py=ply2las.terra_ply2las:main',
            ],
        },

        install_requires=[
            'pika>=0.10.0',
            'requests>=2.11.0',
            'pyclowder'
        ],

        dependency_links=['https://opensource.ncsa.illinois.edu/bitbucket/rest/archive/latest/projects/CATS/repos/pyclowder/archive?format=zip#egg=pyclowder-0.9.2'],

        packages=find_packages(),

        # basic package metadata
        url='https://github.com/terraref/extractors-3dscanner',
        author='Max Burnette',
        author_email='mburnet2@illinois.edu',

        license='NCSA',
        classifiers=[
            'Development Status :: 3 - Alpha',
            'License :: OSI Approved :: NCSA License',

            'Programming Language :: Python :: 2',
            'Programming Language :: Python :: 2.7',
        ],
        keywords='terraref clowder extractor'

)
