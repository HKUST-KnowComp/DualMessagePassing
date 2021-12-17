## Transform

This stage transforms a dataset from its original format to the training input format.

Users need to specify the following parameters in ```transform.sh```:
- **dataset**: choose from ```PubMed``` and ```Yelp```;
- **model**: choose from ```DMPNN```, ```CompGCN```, and ```R-GIN``` (more basedlines can be found [here](https://github.com/yangji9181/HNE/tree/master/Model));
- **attributed**: choose ```False``` for unattributed training;
- **supervised**: choose ```False``` for unsupervised training.

Run ```bash transform.sh``` to complete *Stage 2: Transform*.

We also generate a file including node ids of all labeled nodes and nodes in predicted links.