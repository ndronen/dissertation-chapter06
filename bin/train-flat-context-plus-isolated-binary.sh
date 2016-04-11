#!/bin/bash

# Deep residual network context-dependent model trained jointly with a
# convoutional isolated correction model.
#
# The corpus consists of short words, which offer few opportunities for
# modification by edit patterns learned from a corpus of real errors,
# so we need to inject noise in the right places to avoid overfitting.  
# Those places are (1) the non-word vector that comes from the character
# convolutional block and possibly (2) the cosine similarity of the 
# character convolutional outputs of the non-word and the candidate.

n_train=60000
n_valid=4500
model_dest=models/binary/flat_context_deep_residual_plus_isolated
n_context_embed_dims=20
vectors_path=data/vectors.pkl.gz
pickle_path=data/contexts-100-min-100-max-per-word-length-3.pkl
l2_penalty=0.000001

bin/train.py models/binary \
    --mode persistent-background \
    --model-dest $model_dest \
    --model-cfg fully_connected='[10]' n_residual_blocks=5 n_hidden_residual=10 n_epoch=500 patience=50 samples_per_epoch=$n_train pickle_path=$pickle_path use_context_model=true context_model_type="flat" use_char_model=true use_char_merge=true char_inputs_to_dense_block='[]' batch_normalization=true class_weight_exponent=1 vectors_path=$vectors_path batch_size=100 n_val_samples=$n_valid optimizer=Adam n_context_embed_dims=$n_context_embed_dims train_size=$n_train validation_size=$n_valid l2_penalty=$l2_penalty softmax_max_norm=1 char_merge_gaussian_noise_sd=0.0 non_word_gaussian_noise_sd=0.0 char_input_width=15 char_filter_width=3 n_char_embed_dims=25 n_char_filters=100 only_return_candidates_in_vocabulary=true dropout_residual_p=0.1
