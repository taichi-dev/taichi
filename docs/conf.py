# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import re

'''
html_theme is usually unchanged (rocm_docs_theme).
flavor defines the site header display, select the flavor for the corresponding portals
flavor options: rocm, rocm-docs-home, rocm-blogs, rocm-ds, instinct, ai-developer-hub, local, generic
'''
html_theme = "rocm_docs_theme"
html_theme_options = {"flavor": "rocm-simulation", "repository_url": "https://github.com/ROCm/taichi/"}

'''
docs_header_version is used to manually configure the version in the header. If
there exists a non-null value mapped to docs_header_version, then the header in
the documentation page will contain the given version string.
'''
html_context = {
    "docs_header_version": "25.11"
}


# This section turns on/off article info
setting_all_article_info = True
all_article_info_os = ["linux"]
all_article_info_author = ""

# Dynamically extract component version
with open('../CMakeLists.txt', encoding='utf-8') as f:
    pattern = r'.*\brocm_setup_version\(VERSION\s+([0-9A-Za-z._-]+)' # Update according to each component's CMakeLists.txt
    match = re.search(pattern,
                      f.read())
    if not match:
        raise ValueError("VERSION not found!")
    version_number = match[1]

# for PDF output on Read the Docs
project = "Taichi Lang"
author = "Advanced Micro Devices, Inc."
copyright = "Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved."
version = version_number
release = version_number

external_toc_path = "./sphinx/_toc.yml" # Defines Table of Content structure definition path

# Add more addtional package accordingly
extensions = [
    "rocm_docs", 
    "sphinx.ext.autodoc", # for Python docstrings
] 

html_title = f"{project} {version_number} documentation"

external_projects_current_project = "Taichi Lang"
