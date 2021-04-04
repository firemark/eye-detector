from distutils.core import setup
from setuptools import setup
from pathlib import Path

setup(
    name='eye_detector',
    version='0.1',
    packages=['eye_detector'],
    license='MIT',
    long_description=Path('README.md').read_text(),
    scripts=[
        'scripts/eye_detector',
    ],
)
