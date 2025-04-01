#!/bin/bash
export CUDA_VISIBLE_DEVICES="0"
export RESULT_FOLDER="/data2/wll/checkpoint/mask1/"
python train.py --results_folder $RESULT_FOLDER --data_class microstructure --name model --batch_size 24 --new True --continue_training True --image_size 64 --training_epoch 510 --ema_rate 0.999 --base_channels 32 --save_last False --save_every_epoch 500 --with_attention True --use_text_condition False --use_sketch_condition False --split_dataset False  --lr 1e-4 --optimizier adamw --sdf_folder /data2/wll/mask1_train_data