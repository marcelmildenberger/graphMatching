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
- For improved performance, a CUDA-enabled GPU with \>= 24 GB of VRAM is strongly recommended. Make sure that your GPU has a [compute capability](https://developer.nvidia.com/cuda-gpus) >= 3.7.

If you limit your experiments to smaller datasets (<= 5,000 records), computations can be run on CPU and with around 32 GB of ram. A powerful laptop
should be enough.  

**Note:** While the code itself should be platform independent, we recommend running it on GNU/Linux. The code has been tested on Ubuntu 22.04 LTS only.

#### 0.1. Install System Dependencies
1) Run ``nvidia-smi`` to check if you have a GPU driver installed. If not, install the [latest version](https://www.nvidia.com/download/index.aspx).
2) Make sure that you have ``pkg-config`` and FreeType installed. Otherwise, the installation of matplotlib might fail. On Ubuntu/Debian run ``sudo apt install pkg-config libfreetype6-dev``
3) Install the g++ compiler for improved performance. On Ubuntu run ``sudo apt install g++``
4) Install the [oneAPI Math Kernel Library](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html). On Ubuntu run ``sudo apt install intel-mkl``.

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
There, you will find four dictionaries storing the [configuration and parameters](./parameters.md).
Adjust the parameters to your liking, save the file and run ``main.py``.
If you set the verbose-option to True, detailed status reports will be printed on screen.

___
## How to Reproduce our Results

To reproduce the results we reported in our paper, you may simply run

``python3 benchmark.py``

This will run all experiments reported in Chapter 6 and save the results in the ``./data`` directory.
Benchmark results are tab-separated and contain the results of one experiment per row. The first row is a header,
specifying which values are reported in the respective column. Aside from dumps of the
config dictionaries, the result files include the following performance metrics:

| Value              | Description                                                                                                                           |
|--------------------|---------------------------------------------------------------------------------------------------------------------------------------|
| success_rate       | Number of correctly matched records divided by number of overlapping records.                                                         |
| correct            | Absolute number of correctly matched records.                                                                                         |
| n_alice            | Number of records in Alice's dataset.                                                                                                 |
| n_eve              | Number of records in Eve's dataset.                                                                                                   |
| elapsed_total      | Time elapsed since start of script (seconds)                                                                                          |
| elapsed_alice_enc  | Duration of encoding and similarity calculation (edgeweights) for Alice's data (seconds). Is set to -1 if cached encodings were used. |
| elapsed_eve_enc    | Duration of encoding and similarity calculation (edgeweights) for Eve's data (seconds). Is set to -1 if cached encodings were used.   |
| elapsed_alice_emb  | Duration of calculating embeddings for Alice's data (seconds). Is set to -1 if cached embeddings were used.                           |
| elapsed_eve_emb    | Duration of calculating embeddings for Eve's data (seconds). Is set to -1 if cached embeddings were used.                             |
| elapsed_align_prep | Duration of preparing the data for alignment (seconds). Should be close to 0 unless the *MaxLoad* parameter was set.                  |
| elapsed_align      | Duration of computing alignment (seconds).                                                                                            |
| elapsed_mapping    | Time elapsed since the beginning of the embedding stage (seconds). This is the actual attack duration, as encoding is done by victim. |             


The dumps of the config dictionaries are generated dynamically, i.e. the order
of columns changes based on the order of the keys in the dictionary.

**Note:** Several parts of the attack, most importantly encoding, embedding and
alignment, involve randomness. It is thus extremely unlikely that you are able to
perfectly reproduce our results. However, the overall difference in results should be
negligible.

**Another Note:** Re-Running all experiments will take a considerable amount of time. Depending on your
system specification you might face runtimes in excess of several weeks.


### Reproduce Plots
Once the benchmark is complete, you can generate the result plots used in our paper.
This will require an installation of the programming language [*R*](https://www.r-project.org/).
On Ubuntu, you can install R via

``sudo apt install r-base``

Next, simply generate the plots by running

``Rscript create_plots.R``

which will save the result plots in the ``./plots`` directory as ``.eps`` files. 
