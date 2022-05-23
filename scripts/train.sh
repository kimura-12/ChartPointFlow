#! /bin/bash


cates="airplane"
# cates="chair"
# cates="car"
epochs=15000
latent_dims="256-256"
latent_num_blocks=1
y_dim=12
log_name="results/${cates}/${y_dim}_chart"
python train.py \
        --cates ${cates} \
        --epoch ${epochs} \
        --save_dir ${log_name} \
        --batch_size 128 \
        --lr 2e-3 \
        --n_flow_AF 9 \
        --h_dims_AF 256-256-256 \
        --save_freq 400 \
        --valid_freq 300 \
        --vis_freq 200 \
        --log_freq 1 \
        --use_gumbel True \
        --y_dim ${y_dim} \
        --nonlinearity tanh \
        --temp 1.0e-1 \
        --latent_num_blocks ${latent_num_blocks} \
        --latent_dims ${latent_dims} \
        --train_T True \