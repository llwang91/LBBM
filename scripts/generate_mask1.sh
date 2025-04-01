#!/bin/bash

export CUDA_VISIBLE_DEVICES="1"
python generate.py --step 50 --generate_method generate_based_on_bdr_json --model_path /data2/wll/checkpoint/mask1/epoch=499.ckpt --output_path /data2/wll/mask1/data_sample --json_path /data2/wll/mask1/make_data/nsample.json --bdr_path /data2/wll/mask1/make_data/img/0voxel   --num_batch 8 --batch_idx 0