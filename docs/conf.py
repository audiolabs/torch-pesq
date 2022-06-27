import sys
import os
import sphinx_rtd_theme

extensions = [
    'sphinx.ext.todo',
    'sphinx.ext.autodoc',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosummary',
]

project = 'torch-pesq'
copyright = '2022, International AudioLabs Erlangen'

version = '0.1'
release = '0.1'

html_theme = "sphinx_rtd_theme"
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

latex_documents = [
    ('index', 'pywasn.tex', 'pywasn Documentation',
     'International AudioLabs Erlangen', 'manual'),
]

man_pages = [
    ('index', 'pywasn', 'pywasn Documentation',
     ['International AudioLabs Erlangen'], 1)
]

texinfo_documents = [
    ('index', 'pywasn', 'pywasn Documentation',
     'International AudioLabs Erlangen', 'pywasn',
     'One line description of project.',
     'Miscellaneous'),
]
