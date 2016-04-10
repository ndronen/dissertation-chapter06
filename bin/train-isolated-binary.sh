#!/bin/bash

# This model achieves good performance (F1=0.67) within the first few
# epochs, then begins to overfit.)

bin/train.py models/binary --mode transient --model-cfg fully_connected='[1000,500,100,10,1]' n_char_embed_dims=20 n_char_filters=2000 n_epoch=500 patience=30 samples_per_epoch=20000 pickle_path=data/contexts-100-min-100-max-per-word.pkl use_context_model=false use_char_model=true use_real_word_embedding=false batch_normalization=true class_weight_exponent=1 vectors_path=data/vectors.pkl.gz batch_size=32 char_inputs_to_dense_block='["non_word_output"]' char_merge_gaussian_noise_sd=0.0 non_word_gaussian_noise_sd=0.1 n_val_samples=1000 optimizer=Adam use_non_word_output_mlp=false l2_penalty=0.000001 softmax_max_norm=1 context_model_type="convolutional_context" max_edits=2
