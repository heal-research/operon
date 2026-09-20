from __future__ import annotations

import os
from pathlib import Path

project = "operon"
author = "Operon contributors"
copyright = "2019–present, Operon contributors"

extensions = [
    "breathe",
    "myst_parser",
    "sphinx_immaterial",
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

html_theme = "sphinx_immaterial"
html_theme_options = {
    # Render the complete root toctree in the global sidebar. In particular,
    # this keeps the Guide tree out of the header; navigation.tabs is
    # intentionally not enabled.
    "globaltoc_collapse": False,
    "features": [
        "navigation.expand",
        "search.highlight",
        "search.share",
        "search.suggest",
        "toc.follow",
        "toc.sticky",
    ],
    "palette": [
        {
            "media": "(prefers-color-scheme)",
            "toggle": {
                "icon": "material/brightness-auto",
                "name": "Switch to light mode",
            },
        },
        {
            "media": "(prefers-color-scheme: light)",
            "scheme": "default",
            "primary": "indigo",
            "accent": "blue",
            "toggle": {
                "icon": "material/weather-sunny",
                "name": "Switch to dark mode",
            },
        },
        {
            "media": "(prefers-color-scheme: dark)",
            "scheme": "slate",
            "primary": "indigo",
            "accent": "light-blue",
            "toggle": {
                "icon": "material/weather-night",
                "name": "Switch to system preference",
            },
        },
    ],
}
html_static_path = ["_static"]
html_css_files = ["operon.css"]
