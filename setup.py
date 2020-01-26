import os
from setuptools import setup, find_packages


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="cat_species_classifier",

    description="Classify cat species",

    author="Willy and Liam",

    packages=find_packages(exclude=['data', 'figures', 'output', 'notebooks',
                                   'bin', 'envs', 'tests']),

    long_description=read('README.md'),
)
