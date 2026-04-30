"""
Build qBlock as a module.
"""

from typing import List

from setuptools import find_packages, setup

requires: List[str] = [
    "setuptools==80.9.0",
    "pandas==2.3.3",
    "numpy>=1.23.0,<3.0.0",
    "scipy>=1.10.0,<2.0.0",
]
test: List[str] = ["pytest==9.0.2"]
extra: List[str] = ["pylint==4.0.4", "black==25.12.0", "isort==7.0.0"]

setup(
    name="qBlock",
    version="1.1",
    author="Miłosz Martynow",
    author_email="miloszmartynow@gmail.com",
    description="Block architecture of software for quantum mechanics of matter.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    python_requires="==3.12.*",
    packages=find_packages(),
    install_requires=requires,
    extras_require={
        "test": test,
        "format": ["black==25.12.0", "isort==7.0.0", "pylint==4.0.4"],
        "dev": extra,
        "all": test + extra,
    },
    include_package_data=True,
)
