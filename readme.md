# Graph Matching Attacks Against Privacy-Preserving Record Linkage
___
**THIS IS WORK IN PROGRESS. DOCUMENTATION MAY BE INCOMPLETE, INCORRECT OR OUTDATED!**

This repository contains the accompanying code for the paper

*Revisiting Graph Matching Attacks on Privacy-Preserving Record Linkage*

that is currently in peer review. If you want to replicate or extend our results, please follow the instructions below.
___
## 0. System Requirements
Due do substantially improved performance, we strongly recommend to run the experiments
on a server equipped with a GPU. The larger the datasets you want to run our code on,
the more powerful the system has to be. To fully replicate our results you will need:
- \>= 400 GB of RAM
- \>= 1 TB of disk space (HDD sufficient) if you want to store intermediate results. >= 20 GB of disk space otherwise.
- A CUDA-enabled GPU with \>= 24 GB of VRAM

If you limit your experiments to smaller datasets (<= 5,000 records), a powerful laptop
should be enough. Make sure that your GPU has a [compute capability](https://developer.nvidia.com/cuda-gpus) >= 3.7. 

**Note:** While the code itself should be platform independent, we recommend running it on GNU/Linux.

#### 0.1 Install System Dependencies
1) Run ``nvidia-smi`` to check if you have a GPU driver installed. If not, install the [latest version](https://www.nvidia.com/download/index.aspx).
2) Make sure that you have ``pkg-config`` installed. Otherwise the installation of matplotlib will fail. On Ubuntu/Debian run ``sudo apt install pkg-config``
3) Install the [oneAPI Math Kernel Library](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html). On Ubuntu run ``sudo apt install intel-mkl``.

## 1. Install Python Dependencies
We recommend using a [virtual environment](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/)
for package management to avoid version conflicts.
1) Install [PyTorch](https://pytorch.org/get-started/locally/). Choose the latest (highest) CUDA version. 
2) Run ``pip install -r requirements.txt`` to install the remaining dependencies.
3) [Verify](https://pytorch.org/get-started/locally/#linux-verification) your install.


## 2. Create Dataset
Prepare your dataset by running ``preprocessing.py``. The script will guide you through the process.
If you don't have any suitable datasets yet, you may use the ones from the paper:
- [Fake Name Generator](https://www.fakenamegenerator.com/order.php)
- [Titanic Passenger List](https://en.wikipedia.org/wiki/Passengers_of_the_Titanic#Passenger_list)
- [Euro Census](https://wayback.archive-it.org/12090/20231221144450/https://cros-legacy.ec.europa.eu/content/job-training_en)
- [North Carolina Voter Registry](https://www.ncsbe.gov/results-data/voter-registration-data)

## 3. Run the Experiment
Open the file ``main.py`` in a text editor of your choice. Scroll down to the bottom of the file.
There, you will find four dictionaries storing the configuration and parameters.
Adjust the parameters to your liking, save the file and run ``main.py``.
If you set the verbose-option to True, detailed status reports will be printed on screen.