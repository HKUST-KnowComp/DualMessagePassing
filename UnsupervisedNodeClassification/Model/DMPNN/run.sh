#!/bin/bash

gpu=1
attributed="False"
supervised="False"
negative_sample=5
dropout=0.2
n_hidden=50
n_epochs=50 # the epoch here is different with the epoch in original HNE
graph_batch_size=10000
sample_depth=3
sample_width=10
label_batch_size=64
grad_norm=1.0
sampler=randomwalk

for dataset in "PubMed" "Yelp"
do
    folder="data/${dataset}/"
    node_file="${folder}node.dat"
    label_file="${folder}label.dat"
    link_file="${folder}link.dat"
    for lr in 1e-2 1e-3
    do
        for reg in 1e-2 1e-3
        do
        for n_layers in 1 2
            do
                for graph_split_size in 0.5 0.7 0.9
                do
                    emb_file="${folder}emb_noattr_unsup_${sampler}_lr${lr}_reg${reg}_nlayer${n_layers}_gsplit${graph_split_size}_hidden${n_hidden}.dat"
                    record_file="${folder}record_noattr_unsup_${sampler}_lr${lr}_reg${reg}_nlayer${n_layers}_gsplit${graph_split_size}_hidden${n_hidden}.dat"
                    OMP_NUM_THREADS=4 python src/main.py \
                        --link ${link_file} \
                        --node ${node_file} \
                        --label ${label_file} \
                        --output ${emb_file} \
                        --n-hidden ${n_hidden} \
                        --negative-sample ${negative_sample} \
                        --lr ${lr} \
                        --dropout ${dropout} \
                        --gpu ${gpu} \
                        --n-layers ${n_layers} \
                        --n-epochs ${n_epochs} \
                        --regularization ${reg} \
                        --grad-norm ${grad_norm} \
                        --graph-batch-size ${graph_batch_size} \
                        --graph-split-size ${graph_split_size} \
                        --label-batch-size ${label_batch_size} \
                        --sampler ${sampler} \
                        --sample-depth ${sample_depth} \
                        --sample-width ${sample_width} \
                        --attributed ${attributed} \
                        --supervised ${supervised}
                done
            done
        done
    done
done

