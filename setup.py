from setuptools import setup, find_packages

name = "thesis"
version = "0.1.0"
description = "A package for simulating and analyzing communication systems."
author = "Zach Myers"

setup(
    name=name,
    version=version,
    description=description,
    author=author,
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "seaborn",
        "pandas"
    ],
    entry_points={
        'console_scripts': [
            'thesis = thesis.__main__:main'
        ]
    },
)