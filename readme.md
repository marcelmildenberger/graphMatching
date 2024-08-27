# Graph Matching Attacks Against Privacy-Preserving Record Linkage
___
**THIS IS WORK IN PROGRESS. DOCUMENTATION MAY BE INCOMPLETE, INCORRECT OR OUTDATED!**

This repository contains the accompanying code for the paper

*Revisiting Graph Matching Attacks on Privacy-Preserving Record Linkage*

that is currently under peer review. If you want to replicate or extend our results, please follow the instructions below.
___
## System Requirements
Due do substantially improved performance, we strongly recommend to run the experiments
on a server equipped with a GPU. The larger the datasets you want to run our code on,
the more powerful the system has to be. To fully replicate our results you will need:
- \>= 400 GB of RAM
- \>= 1 TB of disk space (HDD sufficient) if you want to store intermediate results. >= 20 GB of disk space otherwise.
- For improved performance, a CUDA-enabled GPU with \>= 24 GB of VRAM is strongly recommended. Make sure that your GPU has a [compute capability](https://developer.nvidia.com/cuda-gpus) >= 3.7.

If you limit your experiments to smaller datasets (<= 5,000 records), computations can be run on CPU and with around 32 GB of ram. A powerful laptop
should be enough.  
___
## Set up the Environment
We recommend running our code in a Docker container, as this means you won't have to worry about
dependencies or version conflicts.
If you prefer to run the code directly on your machine, you can find instructions
for a bare-metal install [here.](./docs/plain_install.md)

1) Install Docker by following the [official instructions](https://docs.docker.com/get-started/get-docker/).
2) Open a terminal, navigate to this repository and run ``docker build -t gma:1.0 ./``.
3) Wait a few minutes while Docker builds the image.
4) Run ``docker run -it --gpus all --name gma-artifact gma:1.0`` in your terminal.
5) That's it!

If everything worked, you should now see a shell connected to your docker container. It looks somethin like

``root@bfeff35dda4a:/usr/app# ``

Type ``ls`` to view the contents of the directory you're currently in. The output should look like this:
````angular2html
'Benchmark Results'   aligners       docs        main.py    preprocessing.py   utils.py
 Dockerfile           benchmark.py   embedders   matchers   readme.md          __pycache__
 data                 encoders       requirements.txt
 ````
You can interact with the docker container just like with any other Linux OS.

**Note:** ``docker run`` will always create a new docker container, so you do not have access
to any files you created in previous runs. Use ``docker start -i gma-artifact`` instead to
resume working with the container you already created.

**A note for Windows users:** Make sure to select WSL2 as the subsystem for Docker, otherwise
you won't be able to use the GPU.
___
## How to Run the Code
You can run your own experiments by editing the ``main.py`` file. To do so, open the file in a text editor like [Nano](https://linuxize.com/post/how-to-use-nano-text-editor/#opening-and-creating-files): ``nano main.py``.
Scroll down to the bottom of the file.
There, you will find four dictionaries storing the [configuration and parameters](./docs/parameters.md).
Adjust the parameters to your liking, save the file and start the experiment via ``python main.py``.
If you set the verbose-option to True, detailed status reports will be printed on screen.

To exit the docker container, simply type ``exit`` into the terminal and hit enter.


## 2. Create Dataset
Prepare your dataset by running ``preprocessing.py``. The script will guide you through the process.
If you don't have any suitable datasets yet, you may use the ones from the paper:
- [Fake Name Generator](https://www.fakenamegenerator.com/order.php)
- [Titanic Passenger List](https://en.wikipedia.org/wiki/Passengers_of_the_Titanic#Passenger_list)
- [Euro Census](https://wayback.archive-it.org/12090/20231221144450/https://cros-legacy.ec.europa.eu/content/job-training_en)
- [North Carolina Voter Registry](https://www.ncsbe.gov/results-data/voter-registration-data)

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
Simply generate the plots by running

``Rscript create_plots.R``

which will save the result plots in the ``./plots`` directory as ``.eps`` files. 
