# Running the code without Docker

While the code itself should be platform independent, we recommend running it on GNU/Linux. The code has been tested on Ubuntu 22.04 LTS only.


## 1. Install System Dependencies
1) Run ``nvidia-smi`` to check if you have a GPU driver installed. If not, install the [latest version](https://www.nvidia.com/download/index.aspx).
2) Install Python and pip if necessary. On Ubuntu/Debian run ``sudo apt install python3 python3-pip``.
3) Make sure that you have ``pkg-config`` and FreeType installed. Otherwise, the installation of matplotlib might fail. On Ubuntu/Debian run ``sudo apt install pkg-config libfreetype6-dev``.
4) Install the g++ compiler for improved performance. On Ubuntu run ``sudo apt install g++``.
5) Install the [oneAPI Math Kernel Library](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html). On Ubuntu run ``sudo apt install intel-mkl``.
6) Install [R](https://www.r-project.org/). On Ubuntu run ``sudo apt install r-base``.

## 2. Install Python Dependencies
We recommend using a [virtual environment](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/)
for package management to avoid version conflicts.
1) Install [PyTorch](https://pytorch.org/get-started/locally/). Choose the latest (highest) CUDA version. 
2) Run ``pip install -r requirements.txt`` to install the remaining dependencies.
3) [Verify](https://pytorch.org/get-started/locally/#linux-verification) your install.

## 3 Install R Dependencies
1) Run ``Rscript -e 'install.packages(c("lubridate", "dplyr", "tidyr", "ggplot2"))'``
