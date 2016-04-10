#!/bin/bash

# This script has the command to train a good context-only (i.e. lexical
# context of non-word and a candidate) binary model using only a word
# embedding layer and a deep residual network.  The data are words of
# length 3.  The training examples are windows of width 5 from sentences
# taken from Wikipedia; the word at the center of the window a word of
# length 3 that occurs in the Aspell English dictionary.

n_train=60000
n_valid=4500
model_dest=models/binary/flat_context_deep_residual
vectors_path=data/vectors.pkl.gz
pickle_path=data/contexts-100-min-100-max-per-word-length-3.pkl
l2_penalty=0.000001
n_hidden=10
n_residual_blocks=10
n_context_embed_dims=20

bin/train.py models/binary \
    --mode persistent-background \
    --model-dest $model_dest \
    --model-cfg fully_connected='[$n_hidden]' n_residual_blocks=$n_residual_blocks n_hidden_residual=$n_hidden n_epoch=500 patience=50 samples_per_epoch=$n_train pickle_path=$pickle_path use_context_model=true context_model_type="flat" use_char_model=false batch_normalization=true class_weight_exponent=1 vectors_path=$vectors_path batch_size=100 n_val_samples=$n_valid optimizer=Adam n_context_embed_dims=$n_context_embed_dims train_size=$n_train validation_size=$n_valid l2_penalty=$l2_penalty 
