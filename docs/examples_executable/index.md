# Executable Examples

This section contains comprehensive, executable examples that demonstrate the full capabilities of the incline package. These examples use jupyter-sphinx to run code during documentation build time, generating static plots and tables for GitHub Pages.

## Available Examples

```{toctree}
:maxdepth: 1

basic_usage
advanced_methods
```

## About These Examples

All code examples in this section are executed during documentation build using jupyter-sphinx. This means:

- **Static Output**: Generated plots and tables are included in the final documentation
- **Always Current**: Examples run against the latest code during build
- **No Runtime Dependencies**: Readers don't need Jupyter installed to view results
- **GitHub Pages Compatible**: Works perfectly with static site hosting

## Getting Started

If you're new to incline, start with the [Basic Usage Examples](basic_usage.md) which cover:
- Core trend estimation methods
- Parameter selection guidelines
- Performance comparisons
- Real-world data handling

For more sophisticated analysis, see the [Advanced Methods Examples](advanced_methods.md) which demonstrate:
- Gaussian Process trend estimation
- Kalman filter tracking
- Seasonal decomposition
- Multiscale analysis (SiZer)

## Running Examples Locally

To run these examples in your own environment:

```bash
# Install incline with example dependencies
pip install incline[advanced]

# Or install from source with docs dependencies
pip install -e .[docs]

# Run Jupyter to explore interactively
jupyter notebook
```