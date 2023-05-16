# RadComp: Radial compressor mean-line model

Mean-line (1D) model for evaluating radial compressors. The code is adapted
from the version developed by Schiffmann and Favrat[^1], and was used to generate
a turbo-compressor dataset for *DATED*.

Cyril Picard, Jürg Schiffmann and Faez Ahmed, "DATED: Guidelines for Creating Synthetic
Datasets for Engineering Design Applications", 2023.

[^1]: Jürg Schiffmann and Daniel Favrat, “Design, experimental investigation and multi-objective optimization of a small-scale radial compressor for heat pump applications,” Energy, vol. 35, no. 1, pp. 436–450, Jan. 2010, doi: [10.1016/j.energy.2009.10.010](https://doi.org/10.1016/j.energy.2009.10.010).


## Datasets

The dataset related to *DATED* will be uploaded soon.

## Using the model

### Installation

To install this package, first make sure that you have Python >= 3.9 installed in your environment. If not, we
recommend to install Python using [mambaforge](https://github.com/conda-forge/miniforge#mambaforge).

```bash
git clone https://github.com/cyrilpic/radcomp
cd radcomp
pip install .
```

If you want to install all dependencies to use the dataset generation scripts:

```bash
pip install .[generate]
```

### Basic Usage

A step-by-step example is provided in the [EvaluateCompressor.ipynb notebook](notebooks/EvaluateCompressor.ipynb).


## Citation

If you use the dataset or the model, you can cite our publication:

Cyril Picard, Jürg Schiffmann and Faez Ahmed, "DATED: Guidelines for Creating Synthetic
Datasets for Engineering Design Applications", 2023.
