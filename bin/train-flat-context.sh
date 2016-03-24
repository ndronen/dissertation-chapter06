#!/bin/bash -xe

bin/train.py models/binary --mode transient \
    --model-cfg min_edits=1 max_edits=1 fully_connected=[500,100,10,5] samples_per_epoch=20000 n_epoch=50 patience=5 n_char_filters=1000 n_char_embed_dims=20 n_context_embed_dims=200 class_weight_exponent=1 model_type=flat_context pickle_path=data/contexts-100-min-100-max-per-word.pkl char_merge_mode=cos n_random_candidates=0 n_train_neighbors=0 optimizer=Adam n_residual_blocks=3 n_hidden_residual=5 non_word_generator=LearnedEditTransformer char_merge_act=tanh scale_char_merge_output=true gaussian_noise_sd=0.001 batch_normalization=true
