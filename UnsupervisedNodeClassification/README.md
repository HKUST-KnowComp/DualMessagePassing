# Unsupervised Node Classification

This part is modified from [HNE](https://github.com/yangji9181/HNE)

## Reproduction

### Stage 1: Data

We conduct experiments on 2 HIN benchmark datasets: ```PubMed``` and ```Yelp```.
Please refer to the ```Data``` folder for more details.

### Stage 2: Transform

This stage transforms a dataset from its original format to the training input format.

Users need to specify the targeting dataset, the targeting model, and the training settings.

Please refer to the ```Transform``` folder for more details.

### Stage 3: Model

We add ```DMPNN``` and 2 more heterogeneous Message-Passing baseline implementaions (```CompGCN``` and ```R-GIN```)

Please refer to the ```Model``` folder for more details.

### Stage 4: Evaluate

This stage evaluates the output embeddings based on specific tasks. 

Users need to specify the targeting dataset, the targeting model, and the evaluation tasks.

Please refer to the ```Evaluate``` folder for more details.
