# Installation

## Requirements

incline requires Python 3.8 or later and the following packages:

- pandas >= 1.0.0
- numpy >= 1.18.0
- scipy >= 1.4.0

## Install from PyPI

The easiest way to install incline is via pip:

```bash
pip install incline
```

## Install from Source

You can also install incline from source:

```bash
git clone https://github.com/finite-sample/incline.git
cd incline
pip install -e .
```

## Development Installation

If you want to contribute to incline or modify the source code:

```bash
git clone https://github.com/finite-sample/incline.git
cd incline
pip install -e .[dev,test]
```

This will install incline in development mode with all the development dependencies including testing and documentation tools.

## Verify Installation

To verify that incline is installed correctly, you can run:

```python
import incline
print(incline.__version__)
```

Or test with a simple example:

```python
from incline import naive_trend
import pandas as pd

df = pd.DataFrame({'value': [1, 2, 3, 4, 5]})
result = naive_trend(df)
print(result)
```