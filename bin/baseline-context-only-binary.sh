#!/bin/bash

# This script will eventually have the command to train
# a good context-only (i.e. lexical context of non-word
# and a candidate) binary model.

# TODO: ensure that when a non-word is generated from a real word,
# that the non-word is not in the vocabulary.

bin/train.py models/binary --mode transient --model-cfg fully_connected='[1000,300,100,30,10]' n_epoch=50 patience=50 samples_per_epoch=20000 pickle_path=data/contexts-100-min-100-max-per-word.pkl use_context_model=true use_char_model=false use_real_word_embedding=true batch_normalization=true class_weight_exponent=1 vectors_path=data/vectors.pkl.gz batch_size=128 char_inputs_to_dense_block='[]' char_merge_gaussian_noise_sd=0.0 n_val_samples=1000 optimizer=Adam softmax_max_norm=1
