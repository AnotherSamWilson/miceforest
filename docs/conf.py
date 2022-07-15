# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
import sphinx
from sphinx.errors import VersionRequirementError
sys.path.insert(0, os.path.abspath('..'))


# -- Project information -----------------------------------------------------

project = 'miceforest'
copyright = '2021, Samuel Von Wilson'
author = 'Samuel Von Wilson'

# The full version, including alpha/beta/rc tags
release = '2021-08-21'

# If your documentation needs a minimal Sphinx version, state it here.
needs_sphinx = '4.2.0'  # Due to sphinx.ext.napoleon, autodoc_typehints
if needs_sphinx > sphinx.__version__:
    message = f'This project needs at least Sphinx v{needs_sphinx}'
    raise VersionRequirementError(message)

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon'
]

autodoc_default_flags = ['members', 'inherited-members', 'show-inheritance']
autodoc_default_options = {
    "members": True,
    "inherited-members": True,
    "show-inheritance": True,
}

# mock out modules
autodoc_mock_imports = [
    'matplotlib',
    'seaborn',
    'numpy',
    'pandas',
    'scipy',
    'scikit-learn',
    'lightgbm'
]

master_doc = 'index'

# hide type hints in API docs
autodoc_typehints = "none"

# Only the class' docstring is inserted.
autoclass_content = 'class'

# Generate autosummary pages.
autosummary_generate = ['ImputationKernel.rst', "utils.rst"]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = 'sphinx_rtd_theme'

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
html_theme_options = {
    'includehidden': False,
    'logo_only': True,
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']


def setup(app):
    app.add_css_file('themes.css')
