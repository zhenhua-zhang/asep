#!/usr/bin/env python
'''Package installation script.'''
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="asep-pkg-zzhua",
    version="0.1.0",
    author="Zhenhua Zhang",
    author_email="zhenhua.zhang217@gmail.com",
    description="A tool to predict allele-specific expression",
    long_description=long_description,
    usrl="https://github.com/zhenhua-zhang/asep",
    packages=["asep"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "scipy == 1.1.0",
        "numpy == 1.15.3",
        "pandas == 0.23.4",
        "joblib == 1.2.0",
        "matplotlib == 3.0.0",
        "scikit-learn == 0.20.0",
        "imbalanced-learn == 0.4.0",
    ],
    python_requires=">=3.5"
)
