#!/bin/bash

# CLIP Training Script for CWRU Dataset
python main.py \
    --data_name "MFPT" \
    --model_dir "./models" \
    --batch_size 4 \
    --epochs 40 \
    --lr 1e-5 \
    --lamb 0.003 \
    --vision_encoder "ViT-L/14" \
    --normlizetype "-1-1" \
    --con_index 2 \
    --adaptive True \
    --momentum 0.9 \
    --rho 0.0001 \
    --weight_decay 0.0005 \
    --gpu_id 0

    