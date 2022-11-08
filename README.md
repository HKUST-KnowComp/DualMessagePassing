# Graph Convolutional Networks with Dual Message Passing for Subgraph Isomorphism Counting and Matching

This repository is an official implementation of the paper Graph Convolutional Networks with Dual Message Passing for Subgraph Isomorphism Counting and Matching.

## Introduction

We propose dual message passing neural networks (DMPNNs) to enhance the substructure representation learning in an asynchronous way for subgraph isomorphism counting and matching as well as unsupervised node classification. 

## Reproduction

### Package Dependencies
* tqdm
* numpy
* pandas
* scipy
* numba >= 0.54.0
* python-igraph == 0.9.11
* torch >= 1.7.0
* dgl >= 0.6.0

Please refer to `SubgraphCountingMatching` and `UnsupervisedNodeClassification`


### Citation
```bibtex
@inproceedings{liu2022graph,
  author    = {Xin Liu, Yangqiu Song},
  title     = {Graph Convolutional Networks with Dual Message Passing for Subgraph Isomorphism Counting and Matching},
  booktitle = {AAAI},
  year      = {2022}
}
```

### Miscellaneous
Please send any questions about the code and/or the algorithm to <xliucr@cse.ust.hk>.
