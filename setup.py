from setuptools import setup, find_packages

install_requires = [
    "numpy>=1.24.1",
    "matplotlib>=3.6.2",
    "scipy>=1.10.1",
    "dask>=2023.7.1",
]

setup(name='2stepRichtPyer', version='1.0', packages=find_packages(), install_requires=install_requires)