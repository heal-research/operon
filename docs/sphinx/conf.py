from __future__ import annotations

import os
from pathlib import Path

project = "operon"
author = "Operon contributors"
copyright = "2019–present, Operon contributors"

extensions = [
    "breathe",
    "myst_parser",
    "sphinxcontrib.mermaid",
]

myst_enable_extensions = ["colon_fence"]
exclude_patterns = ["_build"]

# CMake provides the generated XML location. The fallback lets Read the Docs
# use the XML prepared by its documentation-only CMake build.
doxygen_xml_dir = Path(
    os.environ.get(
        "OPERON_DOXYGEN_XML_DIR",
        Path(__file__).parent / "_build" / "docs" / "xml",
    )
)
breathe_projects = {"operon": str(doxygen_xml_dir)}
breathe_default_project = "operon"

html_theme = "pydata_sphinx_theme"
html_context = {"default_mode": "auto"}
html_theme_options = {
    "navbar_center": [],
    "navbar_end": ["theme-switcher"],
    "show_nav_level": 2,
}
html_static_path = ["_static"]
html_css_files = ["operon.css"]
html_js_files = ["diagram-size.js"]
mermaid_output_format = "raw"
mermaid_init_js = """
mermaid.initialize({
  startOnLoad: true,
  flowchart: {
    useMaxWidth: false,
    nodeSpacing: 24,
    rankSpacing: 32,
    padding: 10
  },
  themeVariables: {
    fontSize: "13px"
  }
});
"""
