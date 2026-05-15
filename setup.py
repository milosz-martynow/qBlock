"""
Build qBlock as a module.
"""

from pathlib import Path
from typing import List

from setuptools import find_packages, setup

PROJECT_NAME: str = "qBlock"
PROJECT_VERSION: str = "1"
PROJECT_AUTHOR: str = "Miłosz Martynow"
PROJECT_AUTHOR_EMAIL: str = "miloszmartynow@gmail.com"
PROJECT_DESCRIPTION: str = (
    "Easy in maintenance software to study electronic structure of atomic "
    "systems via SCF process, including Hartree-Fock and Density Functional "
    "Theory."
)

requires: List[str] = [
    "setuptools==80.9.0",
    "pandas==2.3.3",
    "numpy>=1.23.0,<3.0.0",
    "scipy>=1.10.0,<2.0.0",
]
test: List[str] = ["pytest==9.0.2"]
extra: List[str] = ["pylint==4.0.4", "black==25.12.0", "isort==7.0.0"]

setup(
    name=PROJECT_NAME,
    version=PROJECT_VERSION,
    author=PROJECT_AUTHOR,
    author_email=PROJECT_AUTHOR_EMAIL,
    description=PROJECT_DESCRIPTION,
    long_description=(Path(__file__).parent / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    python_requires="==3.12.*",
    packages=find_packages(),
    package_data={
        "q_block.compute.environment.constants.numerical": ["basis_set/pople/*.gbs"],
        "q_block.learn.examples": [".qblock.config.example"],
    },
    install_requires=requires,
    extras_require={
        "test": test,
        "format": ["black==25.12.0", "isort==7.0.0", "pylint==4.0.4"],
        "dev": extra,
        "all": test + extra,
    },
    include_package_data=True,
)
