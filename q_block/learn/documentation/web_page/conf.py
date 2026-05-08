"""
Sphinx configuration for qBlock documentation.
"""

import sys
from pathlib import Path
from typing import List

_ROOT: Path = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_ROOT))

from setup import PROJECT_AUTHOR, PROJECT_NAME, PROJECT_VERSION

# -- Project information -----------------------------------------------------

project: str = PROJECT_NAME
author: str = PROJECT_AUTHOR
release: str = PROJECT_VERSION
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
]
autoapi_python_class_content: str = "both"

# -- Napoleon (NumPy/Google docstrings support) -------------------------------

napoleon_use_rtype: bool = True
napoleon_use_param: bool = True

# -- HTML output -------------------------------------------------------------

html_theme: str = "furo"
html_title: str = "qBlock"
html_static_path: List[str] = ["_static"]
