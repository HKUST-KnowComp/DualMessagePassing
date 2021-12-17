# Subgraph Isomorphism Counting and Matching

This part is modified from [NeuralSubgraphCounting](https://github.com/HKUST-KnowComp/NeuralSubgraphCounting)

## Reproduction

### Stage 1: Download

We conduct experiments on 4 subgraph isomorphism benchmark datasets: ```Erdos-Renyi```, ```Regular```, ```Complex```, and ```MUTAG```.

Please download data from [OneDrive](https://hkustconnect-my.sharepoint.com/:f:/g/personal/xliucr_connect_ust_hk/ErzdTZguJnFBok2QKUr3yAYBHaReWOYOOAEca0uGzgBlyQ?e=UTs21h).

### Stage 2: Training

We add ```DMPNN``` and ```CompGCN``` for heterogeneous Message-Passing implementaions.
We also add ```DMPLRP``` and ```LRP``` for local relational pooling implementations.

* In order to add reversed edges, please set `--add_rev True`.
* In order to joint learn counting and matching, please set `--node_pred True --match_weights node` (for graph models), `--edge_pred True --match_weights edge` (for sequence models), or `--node_pred True --edge_pred True --match_weights node,edge` (for CompGCN, DMPNN, and DMPLRP, but no further improvement).

##### For Erdos-Renyi
```bash
python train.py \
    --pattern_dir data/Erdos-Renyi/patterns \
    --graph_dir data/Erdos-Renyi/graphs \
    --metadata_dir data/Erdos-Renyi/metadata \
    --save_data_dir data/Erdos-Renyi/datasets \
    --save_model_dir dumps/Erdos-Renyi \
    --add_rev True \
    --hid_dim 64 --node_pred True --edge_pred False \
    --match_weights node \
    --enc_net Multihot --enc_base 2 \
    --emb_net Equivariant --share_emb_net True \
    --rep_net DMPNN \
    --rep_num_pattern_layers 3 --rep_num_graph_layers 3 \
    --rep_residual True --rep_dropout 0.0 --share_rep_net True \
    --pred_net SumPredictNet --pred_hid_dim 64 --pred_dropout 0.0 \
    --max_npv 4 --max_npe 10 --max_npvl 1 --max_npel 1 \
    --max_ngv 10 --max_nge 48 --max_ngvl 1 --max_ngel 1 \
    --train_grad_steps 1 --train_batch_size 64 \
    --train_log_steps 10 --eval_batch_size 64 \
    --lr 1e-3 --train_epochs 100 \
    --seed 0 --gpu_id 0
```

##### For Regular
```bash
python train.py \
    --pattern_dir data/Regular/patterns \
    --graph_dir data/Regular/graphs \
    --metadata_dir data/Regular/metadata \
    --save_data_dir data/Regular/datasets \
    --save_model_dir dumps/Regular \
    --add_rev True \
    --hid_dim 64 --node_pred True --edge_pred False \
    --match_weights node \
    --enc_net Multihot --enc_base 2 \
    --emb_net Equivariant --share_emb_net True \
    --rep_net DMPNN \
    --rep_num_pattern_layers 3 --rep_num_graph_layers 3 \
    --rep_residual True --rep_dropout 0.0 --share_rep_net True \
    --pred_net SumPredictNet --pred_hid_dim 64 --pred_dropout 0.0 \
    --max_npv 4 --max_npe 10 --max_npvl 1 --max_npel 1 \
    --max_ngv 30 --max_nge 90 --max_ngvl 1 --max_ngel 1  \
    --train_grad_steps 1 --train_batch_size 64 \
    --train_log_steps 10 --eval_batch_size 64 \
    --lr 1e-3 --train_epochs 100 \
    --seed 0 --gpu_id 0
```

##### For Complex
```bash
python train.py \
    --pattern_dir data/Complex/patterns \
    --graph_dir data/Complex/graphs \
    --metadata_dir data/Complex/metadata_withoutloop \
    --save_data_dir data/Complex/datasets \
    --save_model_dir dumps/Complex \
    --add_rev True \
    --hid_dim 64 --node_pred True --edge_pred False \
    --match_weights node \
    --enc_net Multihot --enc_base 2 \
    --emb_net Equivariant --share_emb_net True \
    --rep_net DMPNN \
    --rep_num_pattern_layers 3 --rep_num_graph_layers 3 \
    --rep_residual True --rep_dropout 0.0 --share_rep_net True \
    --pred_net SumPredictNet --pred_hid_dim 64 --pred_dropout 0.0 \
    --max_npv 8 --max_npe 8 --max_npvl 8 --max_npel 8 \
    --max_ngv 64 --max_nge 256 --max_ngvl 16 --max_ngel 16 \
    --train_grad_steps 1 --train_batch_size 512 \
    --train_log_steps 100 --eval_batch_size 512 \
    --lr 1e-3 --train_epochs 100 \
    --seed 0 --gpu_id 0
```

##### For MUTAG
```bash
python train.py \
    --pattern_dir data/MUTAG/patterns \
    --graph_dir data/MUTAG/graphs \
    --metadata_dir data/MUTAG/metadata \
    --save_data_dir data/MUTAG/datasets \
    --save_model_dir dumps/MUTAG \
    --add_rev True \
    --hid_dim 64 --node_pred True --edge_pred False \
    --match_weights node \
    --enc_net Multihot --enc_base 2 \
    --emb_net Equivariant --share_emb_net True \
    --rep_net DMPNN \
    --rep_num_pattern_layers 3 --rep_num_graph_layers 3 \
    --rep_residual True --rep_dropout 0.0 --share_rep_net True \
    --pred_net SumPredictNet --pred_hid_dim 64 --pred_dropout 0.0 \
    --max_npv 4 --max_npe 3 --max_npvl 2 --max_npel 2 \
    --max_ngv 28 --max_nge 66 --max_ngvl 7 --max_ngel 4 \
    --train_grad_steps 1 --train_batch_size 32 \
    --train_log_steps 10 --eval_batch_size 32 \
    --lr 1e-3 --train_epochs 200 \
    --seed 0 --gpu_id 0
```

### Stage 3: Evaluation

```bash
python evaluate.py \
    --pattern_dir data/MUTAG/patterns \
    --graph_dir data/MUTAG/graphs \
    --metadata_dir data/MUTAG/metadata \
    --save_data_dir data/MUTAG/datasets \
    --load_model_dir dumps/MUTAG/DMPNN_SumPredictNet_2021_12_09_14_11_52 \
    --eval_batch_size 64
```
