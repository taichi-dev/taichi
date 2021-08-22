# Installation

:::note
The scripts provided here only work for Linux/macOS.

We need an equivalent bat script for Windows.
:::

Download and install the `conda` matching your OS: https://docs.conda.io/en/latest/miniconda.html.

Run the following commands:

```bash
# Assuming you're in the root directory
$ cd conda

$ conda env create -f conda_env.yaml
$ conda activate taichi-dev

# This will only install the dependencies to the 'taichi-dev' environment.
(taichi-dev) $ python3 -m pip install -r ../requirements_dev.txt
```

# Update `taichi-dev`'s environment variables

```bash
# Change scripts/activate_env_vars.sh

$ conda activate taichi-dev

(taichi-dev) $ ./update_env_vars.sh
```
