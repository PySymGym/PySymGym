import os
import sys

sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("../AIAgent"))
sys.path.insert(0, os.path.abspath("_ext"))

project = "PySymGym"
author = "PySymGym contributors"
copyright = "PySymGym contributors"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "navmap",
]

autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}
autodoc_typehints = "description"
napoleon_numpy_docstring = True
nitpicky = True

nitpick_ignore = [
    ("py:class", "common.utils.T"),
]

autodoc_mock_imports = [
    "aiohttp",
    "attrs",
    "cattrs",
    "dataclasses_json",
    "func_timeout",
    "httplib2",
    "joblib",
    "matplotlib",
    "mlflow",
    "natsort",
    "onnx",
    "onnxruntime",
    "optuna",
    "pandas",
    "psutil",
    "pydantic",
    "pynvml",
    "tabulate",
    "torch",
    "torch_geometric",
    "tqdm",
    "websocket",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}

html_theme = "furo"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
