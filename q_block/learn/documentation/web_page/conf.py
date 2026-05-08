"""
Sphinx configuration for qBlock documentation.
"""

from pathlib import Path
from typing import List

_ROOT: Path = Path(__file__).resolve().parents[4]

# -- Project information -----------------------------------------------------

project: str = "qBlock"
author: str = "Miłosz Martynow"
release: str = "1.0.7"
copyright: str = f"2026, {author}"

# -- General configuration ---------------------------------------------------

extensions: List[str] = [
    "autoapi.extension",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
]

# sphinx-autoapi: automatically document the q_block package
autoapi_dirs: List[str] = [str(_ROOT / "q_block")]
autoapi_ignore: List[str] = [
    "*/tests/*",
    "*/learn/*",
]
autoapi_options: List[str] = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    "subpackages",
    "submodules",
]
autoapi_python_class_content: str = "both"

# -- Napoleon (NumPy/Google docstrings support) -------------------------------

napoleon_use_rtype: bool = True
napoleon_use_param: bool = True

# -- HTML output -------------------------------------------------------------

html_theme: str = "furo"
html_title: str = "qBlock"
html_static_path: List[str] = ["_static"]
