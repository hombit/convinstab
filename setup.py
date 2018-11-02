#!/usr/bin/env python


from setuptools import setup

setup(
    name='accdisk-pi-vertstr',
    version='1.0.0',
    url='http://github.com/hombit/convinstab',
    license='MIT',
    author='Konstantin Malanchev',
    author_email='malanchev@sai.msu.ru',
    description='Semi-analytical solution of accretion disk vertical structure equations',
    package_dir={'': 'radiative_cond_disc'},
    py_modules=['vertstr'],
    install_requires=['numpy>=0.11', 'scipy>=0.17'],
)
