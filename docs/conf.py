"""Sphinx configuration — fleet standard via py-canon, plus repo specifics."""

from py_canon.sphinx import configure

# configure() injects the fleet-standard settings into this module's globals;
# the repo-specific additions below mutate them through the same dict so the
# names are visibly defined rather than appearing out of thin air.
ns = globals()
configure(ns)

# The pages under examples_executable/ run their code at build time.
ns["extensions"] += ["jupyter_sphinx"]

# The registered config key is `jupyter_execute_kwargs`. Two other names were
# set here previously -- `jupyter_execute_notebooks` (a myst-nb option, and
# myst-nb is not installed) and `jupyter_sphinx_execution_options` (not a
# registered name at all). Sphinx ignores unknown keys silently, so the build
# had been running with jupyter-sphinx's defaults: no timeout, and
# allow_errors=True. Every example that raised was rendered as a traceback in
# the published docs instead of failing the build.
jupyter_execute_kwargs = {
    "timeout": 60,
    "allow_errors": False,
}

# uncertainty.md embeds the interactive explorer from _static/.
html_static_path = ["_static"]

# The API pages cross-reference the scientific stack in signatures.
ns["intersphinx_mapping"].update(
    {
        "pandas": ("https://pandas.pydata.org/docs/", None),
        "numpy": ("https://numpy.org/doc/stable/", None),
        "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    }
)

# `__init__` is deliberately not in special-members. Most public types here are
# frozen dataclasses whose __init__ is generated and undocumented, and listing
# it makes autosummary emit it both on the class page and as a member, which
# Sphinx reports as a duplicate object description for every one of them.
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}

# Render "Attributes:" as :ivar: fields rather than standalone attribute
# directives. Most public types here are frozen dataclasses, whose attributes
# autodoc already documents; without this, both render and every one of them
# raises a duplicate-object-description warning.
napoleon_use_ivar = True

# Local `make html` writes into docs/build/; keep stale output out of the
# source scan.
ns["exclude_patterns"] += ["build"]
