"""Build qBlock as a module."""

from pathlib import Path
from typing import List

import setuptools

requires: List[str] = ["setuptools==80.9.0"]
test: List[str] = ["pytest==9.0.2"]
extra: List[str] = ["pylint==4.0.4", "black==25.12.0", "isort==7.0.0"]
all_modules: List[str] = requires + test + extra

setuptools.setup(
    name="qBlock",
    version="1.0",
    author="Miłosz Martynow",
    author_email="miloszmartynow@gmail.com",
    description="Block architecture of software for quantum mechanics of matter.",
    py_modules=["q_block", "data", "tests"],
    scripts=[
        str(Path("./q_block/atom.py")),
        str(Path("./q_block/aufbau_exceptions.py")),
        str(Path("./q_block/atoms_data.py")),
        str(Path("./tests/test_atom.py")),
        str(Path("./tests/verification_data/expected_atom_pure.py")),
        str(Path("./tests/verification_data/expected_atom_empirical.py")),
    ],
    python_requires="==3.12.*",
)
