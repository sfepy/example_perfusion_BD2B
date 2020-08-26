#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# example_poropiezo documentation build configuration file, created by
# sphinx-quickstart on Thu Mar 12 09:03:54 2020.
#
# This file is execfile()d with the current directory set to its
# containing dir.
#
# Note that not all possible configuration values are present in this
# autogenerated file.
#
# All configuration values have a default; values that are commented out
# serve to show the default.

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- General configuration ------------------------------------------------

html_logo = './_static/sfepy_logo.png'

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.intersphinx',
    'sphinx.ext.imgmath',
    'sphinx.ext.extlinks',
    'sphinx_rtd_theme',
]
# extensions = ['sphinx.ext.intersphinx', 'sphinx.ext.mathjax']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'

# General information about the project.
project = u'sfepy_example_perfusion_BD2B'
copyright = u'2020, Jana Turjanicová, Vladimír Lukeš, Eduard Rohan'
author = u'Jana Turjanicová, Vladimír Lukeš, Eduard Rohan'

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.
version = '1.0'
# The full version, including alpha/beta/rc tags.
release = '1.0'

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = None

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False

numfig = True

# -- Options for HTML output ----------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'alabaster'
import sphinx_rtd_theme
html_theme = 'sphinx_rtd_theme'
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
html_context = {
    # Enable the "Edit in GitHub link within the header of each page.
    # 'display_github': True,
    # Set the following variables to generate the resulting github URL for each page.
    # Format Template: https://{{ github_host|default("github.com") }}/{{ github_user }}/{{ github_repo }}/blob/{{ github_version }}{{ conf_py_path }}{{ pagename }}{{ suffix }}
    'github_user': 'turjani',
    'github_repo': 'example_BD2B',
    'menu_links_name': 'Connections',
    'menu_links': [
        ('<i class="fa fa-github fa-fw"></i> Source Code', 'https://github.com/sfepy/example_perfusion_BD2B'),
#        ('<i class="fa fa-file-text fa-fw"></i> The Paper', 'https://doi.org/10.1016/j.camwa.2019.04.004'),
        ('<i class="fa fa-external-link fa-fw"></i> SfePy', 'https://sfepy.org'),
    ],
}
html_copy_source = False
html_show_sourcelink = False

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
# html_theme_options = {}
html_theme_options = {
    'logo_only': True,
    'display_version': False,
    # 'navigation_depth': 4,
}
# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

html_style = 'sfepy.css'

# Custom sidebar templates, must be a dictionary that maps document names
# to template names.
#
# This is required for the alabaster theme
# refs: http://alabaster.readthedocs.io/en/latest/installation.html#sidebars
# html_sidebars = {}

# Output file base name for HTML help builder.
htmlhelp_basename = 'example_perfusion_BD2B'


# -- Options for LaTeX output ---------------------------------------------

latex_elements = {
}

imgmath_latex_preamble = r"""
\usepackage{bm}
\def\Om{\Omega}
\def\sigmab{\boldsymbol{\sigma}}
\def\Ab{\bm{A}}
\def\Db{\bm{D}}
\def\Bb{\bm{B}}
\def\Sb{\bm{S}}
\def\Pb{\bm{P}}
\def\Hb{\bm{H}}
\def\Kb{\bm{K}}
\def\Ib{\bm{I}}
\def\0b{\bm{0}}

\def\Acal{\mathcal{A}}
\def\Bcal{\mathcal{B}}
\def\Scal{\mathcal{S}}
\def\Pcal{\mathcal{P}}
\def\Qcal{\mathcal{Q}}
\def\Hcal{\mathcal{H}}
\def\Kcal{\mathcal{K}}
\def\Mcal{\mathcal{M}}
\def\Acalb{\boldsymbol{\mathcal{A}}}
\def\Bcalb{\boldsymbol{\mathcal{B}}}
\def\Scalb{\boldsymbol{\mathcal{S}}}
\def\Pcalb{\boldsymbol{\mathcal{P}}}
\def\Qcalb{\boldsymbol{\mathcal{Q}}}
\def\Hcalb{\boldsymbol{\mathcal{H}}}
\def\Kcalb{\boldsymbol{\mathcal{K}}}
\def\eb{\bm{e}}
\def\db{\bm{d}}
\def\fb{\bm{f}}
\def\gb{\bm{g}}
\def\hb{\bm{h}}
\def\nb{\bm{n}}
\def\ub{\bm{u}}
\def\vb{\bm{v}}
\def\wb{\bm{w}}
\def\veps{\varepsilon}
\def\vphi{\varphi}
\def\vrho{\varrho}
\def\sigmab{\boldsymbol\sigma}
\def\psib{\boldsymbol\psi}
\def\vthetab{\boldsymbol\vartheta}
\def\thetab{\boldsymbol\theta}
\def\omegab{\boldsymbol\omega}
\def\Pib{\boldsymbol\Pi}
\def\dV{\mbox{d}V}
\def\dS{\mbox{d}S}
\def\eeb#1{\eb\left(#1\right)}
\def\eebz#1{\eb_z\left(#1\right)}
\def\eebx#1{\eb_x\left(#1\right)}
\def\eeby#1{\eb_y\left(#1\right)}

\def\Hspace{\vec{H}_\#^1}
\def\Uspace{\boldsymbol{\mathcal U}}
"""   
