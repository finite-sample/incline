# Executable examples

Sphinx executes the code in these Markdown pages when it builds the
documentation. A code error fails the build; successful output is rendered as
static plots and tables.

## Examples

```{toctree}
:maxdepth: 1

basic_usage
advanced_methods
```

## Where to start

Start with [basic usage](basic_usage.md) for the core estimators, smoothing
parameters, and a comparison on simulated data.

The [advanced examples](advanced_methods.md) cover Gaussian processes,
state-space models, seasonal decomposition, and SiZer.

## Build locally

Run the same warning-as-error build used for release validation:

```bash
uv sync --group docs
uv run sphinx-build -W -b html docs docs/_build/html
```
