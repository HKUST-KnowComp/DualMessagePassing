## Models: Message-Passing

**DMPNN: Graph Convolutional Networks with Dual Message Passing for Subgraph Isomorphism Counting and Matching**
```
@inproceedings{liu2022graph,
  title={Graph convolutional networks with dual message passing for subgraph isomorphism counting and matching},
  author={Liu, Xin and Song, Yangqiu},
  booktitle={AAAI},
  year={2022}
}
```

*Source: https://github.com/HKUST-KnowComp/DualMessagePassing/blob/master/src/rgin.py*

**R-GIN: Neural subgraph isomorphism counting**
```
@inproceedings{liu2020neural,
  title={Neural subgraph isomorphism counting},
  author={Liu, Xin and Pan, Haojie and He, Mutian and Song, Yangqiu and Jiang, Xin and Shang, Lifeng},
  booktitle={SIGKDD},
  pages={1959--1969},
  year={2020}
}
```

*Source: https://github.com/HKUST-KnowComp/NeuralSubgraphCounting/blob/master/src/rgin.py*

**CompGCN: Neural subgraph isomorphism counting**
```
@inproceedings{vashishth2019composition,
  title={Composition-based multi-relational graph convolutional networks},
  author={Vashishth, Shikhar and Sanyal, Soumya and Nitin, Vikram and Talukdar, Partha},
  booktitle={ICLR},
  year={2020}
}
```

*Source: https://github.com/dmlc/dgl/blob/master/examples/pytorch/compGCN/models.py*

**R-GCN: Modeling Relational Data with Graph Convolutional Networks**
```
@inproceedings{schlichtkrull2018modeling,
  title={Modeling relational data with graph convolutional networks},
  author={Schlichtkrull, Michael and Kipf, Thomas N and Bloem, Peter and Van Den Berg, Rianne and Titov, Ivan and Welling, Max},
  booktitle={ESWC},
  pages={593--607},
  year={2018},
  organization={Springer}
}
```

*Source: https://github.com/dmlc/dgl/blob/master/examples/pytorch/rgcn/model.py*

### Deployment

This implementation relies on 2 external packages:
- <a href="https://pytorch.org/">[PyTorch]</a>
- <a href="https://github.com/dmlc/dgl">[DGL]</a>

### Input

*Stage 2: Transform* prepares 3 input files stored in ```data/{dataset}```:
- ```node.dat```: This file is only needed for attributed training, each line is formatted as ```{node_id}\t{node_attributes}``` where entries in ```{node_attributes}``` are separated by ```,```.
- ```link.dat```: The first line specifies ```{number_of_nodes} {number_of_link_types}```. Each folloing line is formatted as ```{head_node_id} {link_type} {tail_node_id}```.
- ```label.dat```: This file is only needed for semi-supervised training. Each line is formatted as ```{node_id}\t{node_label}```.

### Run

Users need to specify the targeting dataset and the set of training parameters in ```run.sh```. <br /> 
Run ```bash run.sh``` to start training.

### Output

This implementation generates 1 output file stored in ```data/{dataset}```:
- ```emb.dat```: The first line specifies the parameters used in training. Each following line describes the id and the embeddings of a node. The id and the embeddings are separated by ```\t```. Entries in the embeddings are separated by ``` ```.