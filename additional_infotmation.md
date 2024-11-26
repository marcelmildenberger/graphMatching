# Additional Information
___
This file contains additional information for our paper that had to be moved
to the paper artifacts due to page limitations.

**Index**
* [Node Representations](#node-representations)
* [Result Files](#result-files)
* [Runtimes](#runtimes)
---
## Node Representations
As mentioned in Section 3.3, the representations of similarity graph nodes as proposed by 
[Vidanage et al.](https://doi.org/10.1145/3340531.3411931) strongly depend on an initial blocking step.
The following tables show the representations of a random node from the encoded similarity graph of the 
Titanic dataset and its counterpart from the plaintext similarity graph. In all tables, the victim and 
attacker blocked the records based on the MinHash signatures of the plaintext $2$-grams. 
The representations of the node from the encoded graph are shown twice: Once for the case that the attacker 
uses the same random seed $r$ for blocking as the victim and once for the case that the attacker's random 
seed is different.
[Table 1](#table-1) shows the data for records that were encoded using BF without diffusion, whereas 
[Table 2](#table-2) deals with TSH encoded records and [Table 3](#table-3) with TMH encoded records.
The features are the same as proposed by Vidanage et al.:

* **Node Freq.:** The number of times the $2$-gram set or encoding appear in the graph. If there are no duplicates, this value is always 1.
* **Node Length:** Number of elements in the $4$-gram set (plaintext) or the integer set (TSH or TMH) or the hamming weight of the bit vector (BF or TMH).
* **Node Degr.:** Number of edges connected to the node.
* **Edge Max:** Maximum edge weight of the edges connected to the node.
* **Edge Min:** Minimum edge weight of the edges connected to the node.
* **Edge Avg:** Average edge weight of the edges connected to the node.
* **Edge Std. Dev.:** Standard deviation of edge weights of the edges connected to the node.
* **Egonet Degr.:** Number of edges connecting the node's ego network to nodes outside the ego network.
* **Egonet Dens.:** Density of the node's ego network.
* **Betw. Centr.:** The node's betweenness centrality.
* **Degr. Centr.:** The node's degree centrality.
* **1-Hop $2^x$:** Number of nodes in the 1-hop neighborhood where $\lfloor log_2(node degree) \rfloor = 2^x$.
* **2-Hop $2^x$:** Number of nodes in the 2-hop neighborhood where $\lfloor log_2(node degree) \rfloor = 2^x$.

A per-feature (column-wise) min-max normalization is applied to all values.

### Table 1
*Node features for the nodes representing the same individual in the similarity graphs for plaintext and BF-encoded records from the Titanic-dataset. Initial blocking is performed using the same and different random seeds $r$.*

|                         | **Node Freq.**    | **Node Length**  | **Node Degr.**   | **Edge Max**     | **Edge Min**     | **Edge Avg**    |                 |                 |
|-------------------------|-------------------|------------------|------------------|------------------|------------------|-----------------|-----------------|-----------------|
| **Encoded, r = 17**     | 1                 | 1                | 0.034            | 0.591            | 0.452            | 0.543           |                 |                 |
| **Plaintext, r = 17**   | 1                 | 1                | 0.034            | 0.483            | 0.333            | 0.402           |                 |                 |
| **Plaintext, r = 4242** | 1                 | 1                | 0.448            | 0.450            | 0.310            | 0.438           |                 |                 |
|                         | **Edge Std.Dev.** | **Egonet Degr.** | **Egonet Dens.** | **Betw. Centr.** | **Degr. Centr.** |                 |                 |                 |
| **Encoded, r = 17**     | 0.283             | 0.733            | 0.016            | 0.016            | 0.034            |                 |                 |                 |
| **Plaintext, r = 17**   | 0.236             | 0.733            | 0.015            | 0.027            | 0.035            |                 |                 |                 |
| **Plaintext, r = 4242** | 0.254             | 0.521            | 0.341            | 0.263            | 0.448            |                 |                 |                 |
|                         | **1-Hop $2^0$**   | **1-Hop $2^1$**  | **1-Hop $2^2$**  | **1-Hop $2^3$**  | **1-Hop $2^4$**  | **1-Hop $2^5$** | **1-Hop $2^6$** | **1-Hop $2^7$** |
| **Encoded, r = 17**     | 0                 | 0                | 0.429            | 0.059            | 0.091            | 0               | 0               | 0               |
| **Plaintext, r = 17**   | 0                 | 0                | 0.375            | 0.067            | 0.087            | 0               | 0               | 0               |
| **Plaintext, r = 4242** | 0                 | 0                | 0.222            | 0.286            | 0.571            | 0.208           | 0               | 0               |
|                         | **2-Hop $2^0$**   | **2-Hop $2^1$**  | **2-Hop $2^2$**  | **2-Hop $2^3$**  | **2-Hop $2^4$**  | **2-Hop $2^5$** | **2-Hop $2^6$** | **2-Hop $2^7$** |
| **Encoded, r = 17**     | 0                 | 0                | 0.163            | 0.115            | 0.244            | 0.087           | 0               | 0.015           |
| **Plaintext, r = 17**   | 0                 | 0                | 0.188            | 0.136            | 0.241            | 0.100           | 0               | 0.028           |
| **Plaintext, r = 4242** | 0.200             | 0.429            | 0.448            | 0.383            | 0.392            | 0.175           | 0               | 0               |

### Table 2
*Node features for the nodes representing the same individual in the similarity graphs for plaintext and TSH-encoded records from the Titanic-dataset. Initial blocking is performed using the same and different random seeds $r$.*

|                         | **Node Freq.**    | **Node Length**  | **Node Degr.**   | **Edge Max**     | **Edge Min**     | **Edge Avg**    |                 |                 |
|-------------------------|-------------------|------------------|------------------|------------------|------------------|-----------------|-----------------|-----------------|
| **Encoded, r = 17**     | 1                 | 1                | 0.035            | 0.480            | 0.316            | 0.394           |                 |                 |
| **Plaintext, r = 17**   | 1                 | 1                | 0.035            | 0.483            | 0.333            | 0.402           |                 |                 |
| **Plaintext, r = 4242** | 1                 | 1                | 0.448            | 0.450            | 0.310            | 0.438           |                 |                 |
|                         | **Edge Std.Dev.** | **Egonet Degr.** | **Egonet Dens.** | **Betw. Centr.** | **Degr. Centr.** |                 |                 |                 |
| **Encoded, r = 17**     | 0.242             | 0.733            | 0.014            | 0.020            | 0.035            |                 |                 |                 |
| **Plaintext, r = 17**   | 0.236             | 0.733            | 0.015            | 0.027            | 0.035            |                 |                 |                 |
| **Plaintext, r = 4242** | 0.254             | 0.521            | 0.341            | 0.263            | 0.448            |                 |                 |                 |
|                         | **1-Hop $2^0$**   | **1-Hop $2^1$**  | **1-Hop $2^2$**  | **1-Hop $2^3$**  | **1-Hop $2^4$**  | **1-Hop $2^5$** | **1-Hop $2^6$** | **1-Hop $2^7$** |
| **Encoded, r = 17**     | 0                 | 0                | 0.375            | 0.071            | 0.087            | 0               | 0               | 0               |
| **Plaintext, r = 17**   | 0                 | 0                | 0.375            | 0.067            | 0.087            | 0               | 0               | 0               |
| **Plaintext, r = 4242** | 0                 | 0                | 0.222            | 0.286            | 0.571            | 0.208           | 0               | 0               |
|                         | **2-Hop $2^0$**   | **2-Hop $2^1$**  | **2-Hop $2^2$**  | **2-Hop $2^3$**  | **2-Hop $2^4$**  | **2-Hop $2^5$** | **2-Hop $2^6$** | **2-Hop $2^7$** |
| **Encoded, r = 17**     | 0                 | 0                | 0.200            | 0.157            | 0.250            | 0.118           | 0.020           | 0               |
| **Plaintext, r = 17**   | 0                 | 0                | 0.188            | 0.136            | 0.241            | 0.100           | 0               | 0.028           |
| **Plaintext, r = 4242** | 0.200             | 0.429            | 0.448            | 0.383            | 0.392            | 0.175           | 0               | 0               |

### Table 3
*Node features for the nodes representing the same individual in the similarity graphs for plaintext and TMH-encoded records from the Titanic-dataset. Initial blocking is performed using the same and different random seeds $r$.*

|                         | **Node Freq.**    | **Node Length**  | **Node Degr.**   | **Edge Max**     | **Edge Min**     | **Edge Avg**    |                 |                 |
|-------------------------|-------------------|------------------|------------------|------------------|------------------|-----------------|-----------------|-----------------|
| **Encoded, r = 17**     | 1                 | 1                | 0.035            | 0.483            | 0.333            | 0.402           |                 |                 |
| **Plaintext, r = 17**   | 1                 | 1                | 0.035            | 0.537            | 0.290            | 0.403           |                 |                 |
| **Plaintext, r = 4242** | 1                 | 1                | 0.448            | 0.450            | 0.310            | 0.438           |                 |                 |
|                         | **Edge Std.Dev.** | **Egonet Degr.** | **Egonet Dens.** | **Betw. Centr.** | **Degr. Centr.** |                 |                 |                 |
| **Encoded, r = 17**     | 0.236             | 0.733            | 0.015            | 0.027            | 0.035            |                 |                 |                 |
| **Plaintext, r = 17**   | 0.328             | 0.733            | 0.015            | 0.015            | 0.035            |                 |                 |                 |
| **Plaintext, r = 4242** | 0.254             | 0.521            | 0.341            | 0.263            | 0.448            |                 |                 |                 |
|                         | **1-Hop $2^0$**   | **1-Hop $2^1$**  | **1-Hop $2^2$**  | **1-Hop $2^3$**  | **1-Hop $2^4$**  | **1-Hop $2^5$** | **1-Hop $2^6$** | **1-Hop $2^7$** |
| **Encoded, r = 17**     | 0                 | 0                | 0.375            | 0.067            | 0.087            | 0               | 0               | 0               |
| **Plaintext, r = 17**   | 0                 | 0                | 0.429            | 0.067            | 0.091            | 0               | 0               | 0               |
| **Plaintext, r = 4242** | 0                 | 0                | 0.222            | 0.286            | 0.571            | 0.201           | 0               | 0               |
|                         | **2-Hop $2^0$**   | **2-Hop $2^1$**  | **2-Hop $2^2$**  | **2-Hop $2^3$**  | **2-Hop $2^4$**  | **2-Hop $2^5$** | **2-Hop $2^6$** | **2-Hop $2^7$** |
| **Encoded, r = 17**     | 0                 | 0                | 0.188            | 0.136            | 0.241            | 0.100           | 0               | 0.028           |
| **Plaintext, r = 17**   | 0                 | 0                | 0.180            | 0.162            | 0.236            | 0.105           | 0.025           | 0               |
| **Plaintext, r = 4242** | 0.2               | 0.429            | 0.448            | 0.383            | 0.392            | 0.175           | 0               | 0               |
___

## Result Files

___
## Runtimes
[Table 4](#table-4) compares the runtimes of the attack as reported by Vidanage et al. and measured in our experiments.
Note that the value for step $1$ reported for our experiments also includes the encoding of the data, which is technically not part of the attack, as the attacker receives already encoded records.

### Table 4
*Comparison of runtimes (minutes) on the Euro dataset for Vidanage et al. and our attack.*

|                       | **Encoding**   | **Step 1**   | **Step 2**   | **Step 3**   | **Total**   |
|-----------------------|----------------|--------------|--------------|--------------|-------------|
|                       | **BF**         | 7.11         | 449.36       | 30.0         | 486.47      |
| **Vidanage et al.**   | **TSH**        | 23.78        | 514.75       | 201.2        | 739.73      |
|                       | **TMH**        | 5.75         | 364.46       | 148.34       | 518.55      |
| --------------------- | -------------- | ------------ | ------------ | ------------ | ----------- |
|                       | **BF**         | 0.50         | 26.65        | 0.07         | 26.22       |
| **Ours**              | **TSH**        | 4.08         | 27.47        | 0.07         | 31.62       |
|                       | **TMH**        | 46.85        | 23.27        | 0.13         | 70.25       |