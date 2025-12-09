"""Build qBlock as a module."""

from pathlib import Path
from typing import List

import setuptools

requires: List[str] = ["setuptools==80.9.0"]
test: List[str] = ["pytest==9.0.2"]
extra: List[str] = ["pylint==4.0.4", "black==25.12.0", "isort==7.0.0"]
all_modules: List[str] = requires + test + extra

main_folder = Path("q_block")
test_folder = Path("tests")

setuptools.setup(
    name="qBlock",
    version="1.0",
    author="Miłosz Martynow",
    author_email="miloszmartynow@gmail.com",
    description="Block architecture of software for quantum mechanics of matter.",
    install_requires=requires,
    test_require=test,
    extras_require={"development": extra, "all": all_modules},
    scripts=[
        str(Path.joinpath(main_folder, "atom.py")),
        str(Path.joinpath(main_folder, "aufbau_exceptions.py")),
        str(Path.joinpath(main_folder, "atoms_data.py")),
        str(Path.joinpath(test_folder, "test_atom.py")),
        str(Path.joinpath(test_folder, "data/expected_atom_pure.py")),
        str(Path.joinpath(test_folder, "data/expected_atom_empirical.py")),
    ],
    python_requires="==3.12.*",
)
