# Graph Matching Attacks Against Privacy-Preserving Record Linkage
___
This repository contains the accompanying code for the paper

[*Revisiting Graph Matching Attacks on Privacy-Preserving Record Linkage*](https://www.openconf.org/acsac2024/modules/request.php?module=oc_program&action=summary.php&id=17)

as presented at ACSAC 2024, Honolulu, Hawaii.
Please follow the instructions below to set up your system.
There are dedicated guides to [reproduce](./docs/reproduction.md) or [extend](./docs/extension.md) our work. Additional information
referenced in the paper can be found [here](./additional_infotmation.md).
___
## System Requirements
Due to substantially improved performance, we strongly recommend to run the experiments
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

If everything worked, you should now see a shell connected to your docker container. It looks something like

``root@bfeff35dda4a:/usr/app# ``

Type ``ls`` to view the contents of the directory you're currently in. The output should look like this:
````
'Benchmark Results'   aligners       docs        gma.py    preprocessing.py   utils.py
 Dockerfile           benchmark.py   embedders   matchers   readme.md          __pycache__
 data                 encoders       requirements.txt
 ````
You can interact with the docker container just like with any other Linux OS.
To exit the docker container, simply type ``exit`` into the terminal and hit enter.

**Note:** ``docker run`` will always create a new docker container, so you do not have access
to any files you created in previous runs. Use ``docker start -i gma-artifact`` instead to
resume working with the container you already created.

**A note for Windows users:** Make sure to select WSL2 as the subsystem for Docker, otherwise
you won't be able to use the GPU.
___
## Prepare your Dataset
The code expects a tab-separated file with one record per row. The fist row must be a 
header specifying the column names.
You may include an arbitrary number of columns. Internally, the values stored in the
columns are concatenated according to column ordering and normalized (switch to lowercase, remove whitespace and missing values).
**The last column must contain a unique ID.**

If you have data in `.csv`, `.xls` or `.xlsx` format, you may run ``python preprocessing.py`` for convenient conversion. 
The script will guide you through the process. 

In the `data` directory, this repository already provides `titanic_full.tsv` which can be used
directly to run the experiments.
___
## Move your Data
Our script expects data to be located in the `./data` directory. Move your prepared datasets there.

If you are using docker, you can copy your data ***to*** the container like this:

``docker cp YOUR_DATA.tsv gma-artifact:/usr/app/data/``

You can also move data, e.g. benchmark results, ***from*** the container:

``docker cp gma-artifact:/usr/app/data/YOUR_DATA.tsv ./``


___
## Run the Code
You can run your own experiments by editing the ``gma.py`` file. To do so, open the file in a text editor like [Nano](https://linuxize.com/post/how-to-use-nano-text-editor/#opening-and-creating-files): ``nano gma.py``.
Scroll down to the bottom of the file.
There, you will find four dictionaries storing the [configuration and parameters](./docs/parameters.md).
Adjust the parameters to your liking, save the file and start the experiment via ``python gma.py``.
If you set the verbose-option to True, detailed status reports will be printed on screen.

___
## Interpret the Results

Upon attack completion, a short status message will be printed:
```
Correct: 9998 of 10000
Success rate: 0.999800
```
This will tell you how many plaintext records were correctly matched to their
encoded counterparts.
If you set the ``BenchMode`` parameter to true, a row with the following, more detailed
information will be added to `./data/benchmark.tsv`.

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

___
## License
This code is licensed under [GPLv3](https://github.com/SchaeferJ/graphMatching/blob/master/LICENSE.txt)