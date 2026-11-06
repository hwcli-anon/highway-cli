"""Compatibility entry point for editable installs with older pip versions."""

from setuptools import find_packages, setup


setup(
    name="highway-decision-cli",
    version="0.4.0rc1",
    description=(
        "Provider-neutral closed-loop decision-model framework for highway "
        "safety testing"
    ),
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=["PyYAML>=6.0"],
    entry_points={"console_scripts": ["highway=highway_cli.cli:main"]},
)
