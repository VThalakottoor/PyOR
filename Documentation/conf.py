# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
import matplotlib
matplotlib.use('agg')  # Force non-interactive backend during build

sys.path.insert(0,os.path.abspath(".."))

project = 'PyOR'
copyright = '2025, Vineeth Francis Thalakottoor Jose Chacko'
author = 'Vineeth Francis Thalakottoor Jose Chacko'
release = '1.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx.ext.todo","sphinx.ext.viewcode","sphinx.ext.autodoc", "nbsphinx","matplotlib.sphinxext.plot_directive"]

#nbsphinx_execute = 'auto'

add_module_names = False

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_logo = "_static/logo.png"


plot_html_show_formats = False  # hide format selector
plot_html_show_source_link = False
plot_formats = [('png', 100)]  # save plots as PNG with 100 dpi


html_baseurl = "https://vthalakottoor.github.io/PyOR/"

html_split_index = False