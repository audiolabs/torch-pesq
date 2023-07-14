from setuptools import setup
from setuptools import find_packages

with open("README.md", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="torch-pesq",
    version="0.1.2",
    description="PyTorch implementation of the Perceptual Evaluation of Speech Quality",
    author="Lorenz Schmidt",
    author_email="lorenz.schmidt@audiolabs-erlangen.de",
    url="https://github.com/audiolabs/torch-pesq",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT",
    install_requires=[
        "numpy",
        "torch",
        "scipy",
        "torchaudio",
        "torchtyping",
        "typeguard==2.13.3",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
    extras_require={
        "tests": [
            "pytest",
            "pytest-black",
            "wheel",
            "pesq",
            "joblib",
        ],
        "docs": [
            "sphinx",
            "sphinx_rtd_theme",
            "myst_parser",
        ],
        "all": [
            "torch-pesq[tests]",
            "torch-pesq[docs]",
        ],
    },
    packages=find_packages(),
)
