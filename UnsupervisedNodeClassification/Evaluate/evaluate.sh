#!/bin/bash

# Note: Only "R-GCN", "R-GIN", "CompGCN", "ConjGCN", "HAN", "MAGNN", and "HGT" support attributed="True" or supervised="True"
# Note: Only "DBLP" and "PubMed" support attributed="True"

attributed="False"
supervised="False"
negative_sample=5
dropout=0.2
n_hidden=50
n_epochs=50 # the epoch here is different with the epoch in original HNE
grad_norm=1.0
sampler=randomwalk

for dataset in "PubMed" "Yelp"
do
    for model in "DMPNN" "CompGCN" "R-GIN" "R-GCN"
    do
        folder="../Model/${model}/data/${dataset}/"
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
                        # record_file="${folder}record_noattr_unsup_${sampler}_lr${lr}_reg${reg}_nlayer${n_layers}_gsplit${graph_split_size}_hidden${n_hidden}.dat"
                        record_file="${folder}record_noattr_unsup_hidden${n_hidden}.dat"
                        OMP_NUM_THREADS=4 python evaluate.py \
                            --dataset ${dataset} \
                            --model ${model} \
                            --task nc \
                            --attributed ${attributed} \
                            --supervised ${supervised} \
                            --emb_file ${emb_file} \
                            --record_file ${record_file}
                    done
                done
            done
        done
    done
    exit 1
done

