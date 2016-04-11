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

model_dest=models/binary/flat_context_deep_residual
vectors_path=data/vectors.pkl.gz
pickle_path=data/contexts-100-min-100-max-per-word.pkl

#n_train=60000
n_train=0.9
n_valid=4500

# Architectural parameters.
n_context_embed_dims=200
char_input_width=15
char_filter_width=4
n_char_filters=100
n_char_embed_dims=25
n_hidden=100
n_residual_blocks=4

# Regularization parameters.
train_context_embeddings=true
l2_penalty=0.000000
dropout_fc_p=0.0
dropout_residual_p=0.0
batch_normalization=false
non_word_gaussian_noise_sd=0.0

bin/train.py models/binary \
    --mode persistent-background \
    --model-dest $model_dest \
    --model-cfg fully_connected="[$n_hidden]" n_residual_blocks=$n_residual_blocks n_hidden_residual=$n_hidden n_epoch=500 patience=50 samples_per_epoch=240000 pickle_path=$pickle_path use_context_model=true context_model_type="flat" use_char_model=true use_char_merge=true char_inputs_to_dense_block='[]' batch_normalization=true class_weight_exponent=1 vectors_path=$vectors_path batch_size=100 n_val_samples=$n_valid optimizer=Adam n_context_embed_dims=$n_context_embed_dims train_size=$n_train validation_size=$n_valid l2_penalty=$l2_penalty softmax_max_norm=1 non_word_gaussian_noise_sd=$non_word_gaussian_noise_sd char_input_width=$char_input_width char_filter_width=$char_filter_width n_char_embed_dims=$n_char_embed_dims n_char_filters=$n_char_filters only_return_candidates_in_vocabulary=true dropout_fc_p=$dropout_fc_p dropout_residual_p=$dropout_fc_p
