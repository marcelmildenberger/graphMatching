# Reproduce our Experiments
These are the steps required to reproduce the results we reported in Chapter 6 and Table 4 (Appendix A) of our paper, including the plots of the success rates.

**Note:** Several parts of the attack, most importantly encoding, embedding and
alignment, involve randomness. It is thus extremely unlikely that you are able to
perfectly reproduce our results. However, the overall difference in results should be
negligible.

**Another Note:** Re-Running all experiments will take a considerable amount of time. Depending on your
system specification you might face runtimes in excess of several weeks.
This is due to the large number of parameter combinations for the *BFD* encodings.
To skip this step, edit line 102 of `benchmark.py` and set it to `diffuse = [False]`.
___
### Obtain Datasets
Make sure that you have all required datasets in the  `./data` directory.
The code expects the following files to be present:

```
fakename_1k.tsv     fakename_2k.tsv     fakename_5k.tsv     fakename_10k.tsv 
fakename_20k.tsv    fakename_50k.tsv    euro_full.tsv       ncvoter.tsv 
titanic_full.tsv
```

You may obtain the Euro and NCVoter datasets from here:
- [Euro Census](https://wayback.archive-it.org/12090/20231221144450/https://cros-legacy.ec.europa.eu/content/job-training_en)
- [North Carolina Voter Registry](https://www.ncsbe.gov/results-data/voter-registration-data)

Remember to [prepare](../readme.md#prepare-your-dataset) the dataset so it fits the correct file format.
___
### Run the Benchmarks
To reproduce the results we reported in our paper, you may simply run

``python3 benchmark.py``

This will run all experiments reported in Chapter 6 and save the results in the ``./data`` directory.
Benchmark results are tab-separated and contain the results of one experiment per row. The first row is a header,
specifying which values are reported in the respective column. Aside from dumps of the
config dictionaries, the result files include detailed performance metrics.


The dumps of the config dictionaries are generated dynamically, i.e. the order
of columns changes based on the order of the keys in the dictionary.
___
### Reproduce Plots
Once the benchmark is complete, you can generate the result plots used in our paper.
Simply generate the plots by running

``Rscript create_plots.R``

which will save the result plots in the ``./plots`` directory as ``.eps`` files. 
___
## Reproduce our Replication Study

You can also reproduce the results of our replication study for the original Graph Matching Attack
by [Vidanage et al.](https://doi.org/10.1145/3340531.3411931). These results have been reported in
Chapter 3.2 of our paper.

You can run our re-implementation of the original through the following command:

`python3 reimplementation_vidanage.py`

You may want to alter the *Disable*, *AliceRandomSeed* and *EveRandomSeed* in the
`BLOCKING_CONFIG` dictionary to re-create the influence blocking and the choice
of random seeds has on the attack performance.