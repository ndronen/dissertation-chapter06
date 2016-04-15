#!/bin/bash

# Deep residual 1-d convolutional network context-dependent model trained
# jointly with a convoutional isolated correction model.
#
# The corpus consists of short words, which offer few opportunities for
# modification by edit patterns learned from a corpus of real errors,
# so we need to inject noise in the right places to avoid overfitting.  
# Those places are (1) the non-word vector that comes from the character
# convolutional block and possibly (2) the cosine similarity of the 
# character convolutional outputs of the non-word and the candidate.

model_dest=models/binary/convolutional_context_deep_residual_plus_isolated/gaussian_sd_0_02_no_caps_500_epoch
mkdir -p $model_dest
vectors_path=null
pickle_path=data/contexts-length34-1000-context-per-word.pkl

#n_train=60000
n_train=0.9
n_valid=4500

# Architectural parameters.
n_context_filters=1000
n_context_embed_dims=50
char_input_width=15
char_filter_width=4
n_char_filters=100
n_char_embed_dims=25
n_hidden=100
n_residual_blocks=4

# Regularization parameters.
train_context_embeddings=true
l2_penalty=0.0
dropout_fc_p=0.0
dropout_residual_p=0.0
batch_normalization=false
non_word_gaussian_noise_sd=0.02
softmax_max_norm=1000
dense_max_norm=1000
residual_max_norm=1000

min_contexts=100
max_contexts=1000
allow_non_word_generator_to_generate_real_words=false
allow_non_word_generator_to_generate_known_non_words=true
allow_non_word_generator_to_generate_caps_words=false
train_only_return_candidates_in_vocabulary=true
exclude_contexts_containing_unknown_words=true

if [ ! -z "$model_dest" ]
then
    echo "copying script to $model_dest"
    echo cp $0 $model_dest/
    cp $0 $model_dest/
fi

    #--mode persistent-background \
    #--model-dest $model_dest \
bin/train.py models/binary \
    --mode transient \
    --model-cfg fully_connected="[$n_hidden]" n_residual_blocks=$n_residual_blocks n_hidden_residual=$n_hidden n_epoch=500 patience=50 samples_per_epoch=240000 pickle_path=$pickle_path use_context_model=true context_model_type="convolutional" use_char_model=true use_char_merge=false use_real_word_embedding=true char_inputs_to_dense_block='["non_word_output", "real_word_output"]' class_weight_exponent=1 vectors_path=$vectors_path batch_size=100 n_val_samples=$n_valid n_context_embed_dims=$n_context_embed_dims train_size=$n_train validation_size=$n_valid l2_penalty=$l2_penalty softmax_max_norm=$softmax_max_norm non_word_gaussian_noise_sd=$non_word_gaussian_noise_sd char_input_width=$char_input_width char_filter_width=$char_filter_width n_char_embed_dims=$n_char_embed_dims n_char_filters=$n_char_filters only_return_candidates_in_vocabulary=true dropout_fc_p=$dropout_fc_p dropout_residual_p=$dropout_residual_p n_context_filters=$n_context_filters char_merge_act=tanh scale_char_merge_output=true activation=relu dense_max_norm=$dense_max_norm residual_max_norm=$residual_max_norm \
    exclude_contexts_containing_unknown_words=$exclude_contexts_containing_unknown_words \
    min_contexts=$min_contexts \
    max_contexts=$max_contexts \
    allow_non_word_generator_to_generate_real_words=$allow_non_word_generator_to_generate_real_words \
    allow_non_word_generator_to_generate_known_non_words=$allow_non_word_generator_to_generate_known_non_words \
    allow_non_word_generator_to_generate_caps_words=$allow_non_word_generator_to_generate_caps_words \
    train_only_return_candidates_in_vocabulary=$train_only_return_candidates_in_vocabulary \
    optimizer=Adam \
    batch_normalization=$batch_normalization
