#!/bin/bash

# This model achieves good performance (F1=0.67) within the first few
# epochs, then begins to overfit.)

#n_train=60000
n_train=0.9
n_valid=4500
model_dest=models/binary/isolated
vectors_path=data/vectors.pkl.gz
pickle_path=data/contexts-100-min-100-max-per-word.pkl
l2_penalty=0.0
n_hidden=100
n_residual_blocks=4
n_char_filters=1000
non_word_gaussian_noise_sd=0.05

    #--mode persistent-background \
    #--model-dest $model_dest \
bin/train.py models/binary \
    --mode transient \
    --model-cfg fully_connected="[$n_hidden]" n_residual_blocks=$n_residual_blocks n_hidden_residual=$n_hidden char_input_width=15 char_filter_width=3 n_char_embed_dims=25 n_char_filters=$n_char_filters n_epoch=500 patience=50 samples_per_epoch=40000 pickle_path=$pickle_path use_context_model=false use_char_model=true use_real_word_embedding=false batch_normalization=true class_weight_exponent=1 vectors_path=null batch_size=32 char_inputs_to_dense_block='["non_word_output"]' non_word_gaussian_noise_sd=$non_word_gaussian_noise_sd n_val_samples=$n_valid optimizer=Adam use_non_word_output_mlp=false l2_penalty=$l2_penalty softmax_max_norm=1000
