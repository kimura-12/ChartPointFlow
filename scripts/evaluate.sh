#! /bin/bash

cates="airplane"
# cates="chair"
# cates="car"
latent_dims="256-256"
latent_num_blocks=1
chart=24

python test.py \
    --cates ${cates} \
    --load_checkpoint pretrained_model/${cates}/${chart}_chart_checkpoint.pt \
    --n_flow_AF 9 \
    --h_dims_AF 256-256-256 \
    --y_dim ${chart} \
    --use_gumbel True \
    --nonlinearity tanh \
    --latent_num_blocks ${latent_num_blocks} \
    --latent_dims ${latent_dims} \
    --train_T True \
    --reconst_eval True \
    --data_dir "../dataset/ShapeNetCore.v2.PC15k"