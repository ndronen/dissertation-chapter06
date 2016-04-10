#!/bin/bash

# TODO: what is the effect of varying the size of the context window?
# TODO: what is the effect of use_real_word_embedding=true?

# This script has the command to train a good context-only (i.e. lexical
# context of non-word and a candidate) binary model.

#n_train=1659738
n_train=100000
#model_dest=models/binary/contextual_01
context_input_width=9
context_filter_width=9
use_real_word_embedding=false
n_context_embed_dims=10
n_context_filters=2000

    #--model-dest $model_dest \
bin/train.py models/binary \
    --mode transient \
    --model-cfg fully_connected='[1000]' n_residual_blocks=5 n_hidden_residual=1000 n_epoch=500 patience=50 samples_per_epoch=$n_train pickle_path=data/contexts-100-min-100-max-per-word.pkl use_context_model=true use_char_model=false use_real_word_embedding=$use_real_word_embedding batch_normalization=true class_weight_exponent=1 vectors_path=data/vectors.pkl.gz batch_size=128 char_inputs_to_dense_block='[]' char_merge_gaussian_noise_sd=0.0 n_val_samples=1000 optimizer=Adam softmax_max_norm=1 l2_penalty=0.00001 n_context_embed_dims=$n_context_embed_dims n_context_filters=$n_context_filters context_input_width=$context_input_width context_filter_width=$context_filter_width