#!/bin/bash

# TODO: what is the effect of varying the size of the context window?
# TODO: what is the effect of use_real_word_embedding=true?

# This script has the command to train a good context-only (i.e. lexical
# context of non-word and a candidate) binary model.

#n_train=1659738
n_train=100000
n_valid=2000
#model_dest=models/binary/contextual_01
context_model_type="flat"
use_real_word_embedding=true
n_context_embed_dims=100

    #--model-dest $model_dest \
bin/train.py models/binary \
    --mode transient \
    --model-cfg fully_connected='[100]' n_residual_blocks=2 n_hidden_residual=100 n_epoch=500 patience=50 samples_per_epoch=$n_train pickle_path=data/contexts-100-min-100-max-per-word.pkl use_context_model=true context_model_type="flat" use_char_model=true use_real_word_embedding=false batch_normalization=true class_weight_exponent=1 vectors_path=data/vectors.pkl.gz batch_size=32 n_val_samples=$n_valid optimizer=Adam n_context_embed_dims=$n_context_embed_dims l2_penalty=0.00001 softmax_max_norm=1
#softmax_max_norm=1 l2_penalty=0.00001 n_context_embed_dims=$n_context_embed_dims n_context_filters=$n_context_filters context_input_width=$context_input_width context_filter_width=$context_filter_width
