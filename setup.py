"""
setup.py — KĦAOS-CORE Python validation stack

Install in development mode:
    pip install -e .

This installs the src/ Python packages so tests and scripts can import them
without adding the repo root to PYTHONPATH manually.
"""

from setuptools import setup, find_packages

setup(
    name="khaos-core",
    version="1.1.0",
    description="Dual-stack BCI architecture with embedded neurorights sovereignty",
    author="KĦAOS-CORE contributors",
    python_requires=">=3.10",
    package_dir={"": "."},
    packages=find_packages(include=["src", "src.*"]),
    install_requires=[
        "numpy>=1.24",
        "scipy>=1.11",
    ],
    extras_require={
        "dashboard": ["matplotlib>=3.7"],
        "hardware":  ["pylsl>=1.16.2"],
        "dev":       ["pytest>=7.4"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
)
