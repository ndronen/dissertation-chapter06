import re
import gzip
import string
import random

from modeling.callbacks import DenseWeightNormCallback
from chapter06.dataset import (
        MulticlassLoader,
        MulticlassSplitter,
        HoldOutByWordSplitter,
        BinaryGenerator,
        ContextTokenizer,
        ContextWindowTransformer,
        MulticlassNonwordTransformer,
        MulticlassContextTransformer,
        RandomWordTransformer,
        LearnedEditTransformer,
        DigitFilter)

from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.utils import check_random_state
from sklearn.neighbors import NearestNeighbors


from spelling.edits import Editor
from spelling.utils import build_progressbar as build_pbar

import sys
# You need to increase the recursion limit to compile many-layered
# residual networks.  I think it's a Theano implementation detail,
# not a Keras one.  I suspect the stack depth limit would not be
# exceeded with many-layered residual networks when using TensorFlow
# as the Keras backend....
sys.setrecursionlimit(5000)
import pickle
import threading

import numpy as np
import pandas as pd

from keras.models import Sequential, Graph
from keras.constraints import maxnorm
from keras.regularizers import l1, l2
from keras.utils import np_utils
from keras.layers.core import Dense, Dropout, Activation, Flatten, Layer, Lambda, Reshape, Merge
from keras.layers.noise import GaussianNoise
from keras.layers.normalization import BatchNormalization
import keras.callbacks

from keras.backend import floatx
import theano.tensor as T

import modeling.data
from modeling.builders import (build_embedding_layer,
    build_convolutional_layer, build_pooling_layer,
    build_dense_layer, build_optimizer, load_weights,
    build_hierarchical_softmax_layer)
from modeling.utils import balanced_class_weights

import spelling.dictionary as spelldict

class Schedule(object):
    def __init__(self, model, epoch_lr_pairs):
        self.model = model
        self.epoch_lr_pairs = epoch_lr_pairs

        assert isinstance(epoch_lr_pairs, list)
        for p in epoch_lr_pairs:
            assert isinstance(p[0], int)
            assert isinstance(p[1], (float, str))

    def schedule(self, epoch):
        learning_rate = self.model.optimizer.lr.get_value().item()

        try:
            if epoch == self.epoch_lr_pairs[0][0]:
                pair = self.epoch_lr_pairs.pop(0)
                expression = pair[1]
                try:
                    learning_rate = float(expression)
                except ValueError as e:
                    try:
                        learning_rate = eval('%f %s' % (learning_rate, expression))
                    except Exception as e:
                        pass
                print('new learning rate %f' % learning_rate)
        except IndexError as e:
            pass

        return learning_rate

class Identity(Layer):
    def get_output(self, train):
        return T.cast(self.get_input(train), floatx())

class TakeMiddleWordEmbedding(Layer):
    def __init__(self, input_width):
        super(TakeMiddleWordEmbedding, self).__init__()
        self.input_width = input_width

    def get_output(self, train):
        X = self.get_input(train)
        return self._get_output(X)

    def _get_output(self, X):
        return X[:, int(self.input_width/2.), :]

    @property
    def output_shape(self):
        input_shape = list(self.input_shape)
        input_shape[1] = 1
        return input_shape

def add_bn_relu(graph, config, prev_layer):
    bn_name = prev_layer + '_bn'
    relu_name = prev_layer + '_relu'
    if config.batch_normalization:
        graph.add_node(BatchNormalization(), name=bn_name, input=prev_layer)
        prev_layer = bn_name
    graph.add_node(Activation(config.activation), name=relu_name, input=prev_layer)
    return relu_name

def build_char_model(graph, config):
    # The character model should have two columns, one for the non-word
    # and one for the real word.  A small amount of Gaussian noise should
    # should be applied to the non-word column right after the character
    # embedding layer.

    # Character-level input for the non-word error.
    graph.add_input(config.non_word_char_input_name,
            input_shape=(config.char_input_width,), dtype='int')

    # Character-level input for the real word suggestion.
    graph.add_input(config.real_word_char_input_name,
            input_shape=(config.char_input_width,), dtype='int')

    char_embedding_layer = build_embedding_layer(config,
            input_width=config.char_input_width,
            n_embeddings=config.n_char_embeddings,
            n_embed_dims=config.n_char_embed_dims,
            dropout=config.dropout_embedding_p)

    graph.add_shared_node(char_embedding_layer,
            name='char_embedding',
            inputs=[config.non_word_char_input_name,
                config.real_word_char_input_name],
            outputs=['non_word_char_embed',
                'real_word_char_embed'])

    char_conv_layer = build_convolutional_layer(config,
            n_filters=config.n_char_filters,
            filter_width=config.char_filter_width)

    inputs = ['non_word_char_embed', 'real_word_char_embed']
    outputs = [re.sub('_embed.*', '_conv', s) for s in inputs]
    graph.add_shared_node(char_conv_layer,
            name='char_conv', inputs=inputs, outputs=outputs)

    non_word_char_prev = 'non_word_char_conv'
    real_word_char_prev = 'real_word_char_conv'

    if config.batch_normalization:
        char_conv_bn = BatchNormalization()
        inputs = [non_word_char_prev, real_word_char_prev]
        outputs = [_+'_bn' for _ in inputs]
        graph.add_shared_node(char_conv_bn,
                name='char_conv_bn',
                inputs=inputs, outputs=outputs)
        non_word_char_prev = outputs[0]
        real_word_char_prev = outputs[1]

    if config.non_word_gaussian_noise_sd > 0.:
        graph.add_node(GaussianNoise(config.non_word_gaussian_noise_sd),
                input=non_word_char_prev,
                name='non_word_char_noise')
        non_word_char_prev = 'non_word_char_noise'

    graph.add_shared_node(Activation(config.activation),
            name='char_conv_act',
            inputs=[non_word_char_prev, real_word_char_prev],
            outputs=['non_word_conv_act', 'real_word_conv_act'])

    char_pool = build_pooling_layer(config,
            input_width=config.char_input_width,
            filter_width=config.char_filter_width)
    graph.add_shared_node(char_pool,
            name='char_pool',
            inputs=['non_word_conv_act', 'real_word_conv_act'],
            outputs=['non_word_conv_pool', 'real_word_conv_pool'])

    non_word_flatten = 'non_word_flatten'

    graph.add_shared_node(Flatten(),
            name='char_flatten',
            inputs=['non_word_conv_pool', 'real_word_conv_pool'],
            outputs=[non_word_flatten, 'real_word_flatten'])


    if config.use_non_word_output_mlp:
        layer_basename = non_word_flatten
        prev_layer = non_word_flatten
        for i in range(2):
            graph.add_node(
                    Dense(config.n_char_filters, W_constraint=maxnorm(1)),
                    input=prev_layer,
                    name='%s%02d' % (layer_basename, i))
            non_word_flatten = '%s%02d' % (layer_basename, i)
            prev_layer = non_word_flatten

    if config.char_merge_mode in ['cos', 'dot']:
        dot_axes = ([1], [1])
    else:
        dot_axes = -1

    char_merge_weight = np.array([[1.]])
    char_merge_bias = np.zeros((1,))
    char_merge_layer = Dense(config.char_merge_n_hidden,
            W_constraint=maxnorm(config.char_merge_max_norm),
            weights=[char_merge_weight, char_merge_bias],
            trainable=config.train_char_merge_layer)
    graph.add_node(char_merge_layer,
            name='char_merge',
            inputs=[non_word_flatten, 'real_word_flatten'],
            merge_mode=config.char_merge_mode,
            dot_axes=dot_axes)

    prev_char_layer = 'char_merge'
    if config.scale_char_merge_output:
        if config.char_merge_act == "sigmoid":
            lambda_layer = Lambda(lambda x: 12.*x-6.)
        elif config.char_merge_act == "tanh":
            lambda_layer = Lambda(lambda x: 6.*x-3.)
        else:
            lambda_layer = Lambda(lambda x: x)
        graph.add_node(lambda_layer,
                name='char_merge_scale', input='char_merge')
        prev_char_layer = 'char_merge_scale'

    non_word_output = 'non_word_flatten'
    real_word_output = 'real_word_flatten'
    char_merge_output = 'char_merge_act'

    graph.add_node(Activation(config.char_merge_act),
            name=char_merge_output,
            input=prev_char_layer)

    return non_word_output, real_word_output, char_merge_output

def build_context_model(graph, config, context_embedding_weights=None):
    if config.context_model_type == 'flat':
        return build_flat_context_model(graph, config, context_embedding_weights)
    elif config.context_model_type == 'convolutional':
        return build_convolutional_context_model(graph, config, context_embedding_weights)
    else:
        raise ValueError("unknown context_model_type %s" % 
                config.context_model_type)

def build_convolutional_context_model(graph, config, context_embedding_weights=None):
    # Word-level input for the context of the non-word error.
    graph.add_input(config.context_input_name,
            input_shape=(config.context_input_width,), dtype='int')

    # Context embedding.
    context_embedding = build_embedding_layer(config,
            input_width=config.context_input_width,
            n_embeddings=config.n_context_embeddings,
            n_embed_dims=config.n_context_embed_dims,
            dropout=config.dropout_embedding_p,
            trainable=config.train_context_embeddings)

    if context_embedding_weights is not None:
        config.logger('setting context embedding weights of shape ' + str(
            context_embedding_weights.shape))
        context_embedding.set_weights([context_embedding_weights])

    graph.add_node(context_embedding,
            name='context_embedding',
            input=config.context_input_name)

    if config.use_real_word_embedding:
        graph.add_node(
                TakeMiddleWordEmbedding(config.context_input_width),
                name='real_word_embedding',
                input='context_embedding')
        graph.add_node(Reshape((config.n_context_embed_dims,)),
                name='real_word_reshape',
                input='real_word_embedding')

    # Context convolution.
    context_conv = build_convolutional_layer(config,
            n_filters=config.n_context_filters,
            filter_width=config.context_filter_width)
    context_conv.trainable = config.train_filters
    graph.add_node(context_conv, name='context_conv', input='context_embedding')
    context_prev_layer = add_bn_relu(graph, config, 'context_conv')

    # Context pooling.
    context_pool = build_pooling_layer(config,
            input_width=config.context_input_width,
            filter_width=config.context_filter_width)
    graph.add_node(context_pool,
            name='context_pool', input=context_prev_layer)
    graph.add_node(Flatten(), name='context_flatten', input='context_pool')

    context_output = 'context_flatten'
    real_word_output = 'real_word_reshape'

    return context_output, real_word_output

def build_flat_context_model(graph, config, context_embedding_weights=None):
    #np.random.seed(config.random_state)

    #graph = Graph()

    #######################################################################
    # Character layers
    #######################################################################
    #non_word_output, real_word_output, char_merge_output = build_char_model(graph, config)

    #######################################################################
    # Word layers
    #######################################################################

    # Word-level input for the context of the non-word error.
    context_embedding_inputs = []
    context_embedding_outputs = []
    context_reshape_outputs = []
    for i in range(1, 6):
        name = '%s_%02d' % (config.context_input_name, i)
        graph.add_input(name, input_shape=(1,), dtype='int')
        context_embedding_inputs.append(name)
        context_embedding_outputs.append(
                'context_embedding_%02d' % i)
        context_reshape_outputs.append(
                'context_reshape_%02d' % i)

    context_embedding = build_embedding_layer(config,
            input_width=1,
            n_embeddings=config.n_context_embeddings,
            n_embed_dims=config.n_context_embed_dims,
            dropout=config.dropout_embedding_p)

    if context_embedding_weights is not None:
        config.logger('setting context embedding weights of shape ' + str(
            context_embedding_weights.shape))
        context_embedding.set_weights([context_embedding_weights])

    graph.add_shared_node(context_embedding,
            name='context_embedding',
            inputs=context_embedding_inputs,
            outputs=context_embedding_outputs)

    graph.add_shared_node(Reshape((config.n_context_embed_dims,)),
        name='context_reshape',
        inputs=context_embedding_outputs,
        outputs=context_reshape_outputs)

    graph.add_node(Identity(),
            name='word_context',
            inputs=context_reshape_outputs,
            merge_mode='concat')

    context_output = 'word_context'

    return context_output, None

def build_model(config, n_classes, context_embedding_weights=None):
    np.random.seed(config.random_state)

    config.logger('build_model context_embedding_weights ' + str(type(context_embedding_weights)))

    graph = Graph()

    #######################################################################
    # Character layers
    #######################################################################

    dense_inputs = []
    softmax_inputs = []

    if config.use_char_model:
        non_word_output, real_word_output, char_merge_output = build_char_model(graph, config)

        if config.char_merge_gaussian_noise_sd > 0.:
            graph.add_node(GaussianNoise(config.char_merge_gaussian_noise_sd),
                    input=char_merge_output,
                    name='char_merge_output_noise')
            char_merge_output = 'char_merge_output_noise'

        for char_output in config.char_inputs_to_dense_block:
            dense_inputs.append(locals()[char_output])

        if config.use_char_merge:
            softmax_inputs.append(char_merge_output)

    #######################################################################
    # Word layers
    #######################################################################

    if config.use_context_model or config.use_real_word_embedding:
        config.logger('passing context_embedding_weights to build_context_model ' +
                str(type(context_embedding_weights)))
        context_output, real_word_output = build_context_model(
                graph, config,
                context_embedding_weights=context_embedding_weights)
        dense_inputs.append(context_output)
        if config.use_real_word_embedding:
            if real_word_output is not None:
                dense_inputs.append(real_word_output)

    if config.merge_mode == 'cos':
        dot_axes = ([1], [1])
    else:
        dot_axes = -1

    # Add some number of fully-connected layers without skip connections.
    prev_layer = None
    for i,n_hidden in enumerate(config.fully_connected):
        layer_name = 'dense%02d' %i
        l = build_dense_layer(config, n_hidden=n_hidden,
                max_norm=config.dense_max_norm)
        if i == 0:
            if len(dense_inputs) == 1:
                graph.add_node(l, name=layer_name,
                        input=dense_inputs[0])
            else:
                graph.add_node(l, name=layer_name,
                        inputs=dense_inputs,
                        merge_mode=config.merge_mode,
                        dot_axes=dot_axes)

            prev_layer = layer_name

            # Only add noise to the first dense layer.
            if config.dense_gaussian_noise_sd > 0.:
                graph.add_node(
                        GaussianNoise(config.dense_gaussian_noise_sd),
                        name=layer_name+'gaussian',
                        input=prev_layer)
                prev_layer = prev_layer +'gaussian'
        else:
            graph.add_node(l, name=layer_name, input=prev_layer)
            prev_layer = layer_name

        if config.batch_normalization:
            graph.add_node(BatchNormalization(), name=layer_name+'bn', input=prev_layer)
            prev_layer = layer_name+'bn'
        graph.add_node(Activation(config.activation), name=layer_name+config.activation, input=prev_layer)
        prev_layer = layer_name+config.activation
        if config.dropout_fc_p > 0.:
            graph.add_node(Dropout(config.dropout_fc_p), name=layer_name+'do', input=prev_layer)
            prev_layer = layer_name+'do'

    # Add sequence of residual blocks.
    for i in range(config.n_residual_blocks):
        # Add a fixed number of layers per residual block.
        block_name = '%02d' % i

        graph.add_node(Identity(), name=block_name+'input', input=prev_layer)
        prev_layer = block_input_layer = block_name+'input'

        try:
            n_layers_per_residual_block = config.n_layers_per_residual_block
        except AttributeError:
            n_layers_per_residual_block = 2

        for layer_num in range(n_layers_per_residual_block):
            layer_name = 'h%s%02d' % (block_name, layer_num)
    
            l = Dense(config.n_hidden_residual, init=config.dense_init,
                    W_constraint=maxnorm(config.residual_max_norm),
                    W_regularizer=l2(config.l2_penalty))
            graph.add_node(l, name=layer_name, input=prev_layer)
            prev_layer = layer_name
    
            if config.batch_normalization:
                graph.add_node(BatchNormalization(), name=layer_name+'bn', input=prev_layer)
                prev_layer = layer_name+'bn'
    
            if i < n_layers_per_residual_block:
                graph.add_node(Activation(config.activation), name=layer_name+config.activation, input=prev_layer)
                prev_layer = layer_name+config.activation
                if config.dropout_residual_p > 0.:
                    graph.add_node(Dropout(config.dropout_residual_p), name=layer_name+'do', input=prev_layer)
                    prev_layer = layer_name+'do'

        graph.add_node(Identity(), name=block_name+'output', inputs=[block_input_layer, prev_layer], merge_mode='sum')
        graph.add_node(Activation(config.activation), name=block_name+config.activation, input=block_name+'output')
        prev_layer = block_input_layer = block_name+config.activation

    if prev_layer is None:
        softmax_inputs.extend(dense_inputs)
    else:
        softmax_inputs.append(prev_layer)

    if len(softmax_inputs) == 1:
        graph.add_node(Dense(n_classes, init=config.dense_init,
            W_constraint=maxnorm(config.softmax_max_norm)),
            name='softmax',
            input=softmax_inputs[0])
    else:
        graph.add_node(Dense(n_classes, init=config.dense_init,
            W_constraint=maxnorm(config.softmax_max_norm)),
            name='softmax',
            inputs=softmax_inputs)

    prev_layer = 'softmax'
    if config.batch_normalization:
        graph.add_node(BatchNormalization(), name='softmax_bn', input='softmax')
        prev_layer = 'softmax_bn'
    graph.add_node(Activation('softmax'), name='softmax_activation', input=prev_layer)
    graph.add_output(name='binary_correction_target', input='softmax_activation')

    load_weights(config, graph)

    optimizer = build_optimizer(config)

    graph.compile(loss={'binary_correction_target': config.loss}, optimizer=optimizer)

    return graph

class MetricsCallback(keras.callbacks.Callback):
    def __init__(self, config, generator, n_samples, callbacks, other_generators={}):
        self.__dict__.update(locals())
        del self.self

    def _set_model(self, model):
        self.model = model
        for cb in self.callbacks:
            cb._set_model(model)

    def compute_metrics(self, generator, name, exhaustive, epoch, logs={}, do_callbacks=False):
        correct = []
        y = []
        y_hat = []
        y_hat_binary = []
        y_hat_dictionary = []
        y_hat_dictionary_binary = []
        counter = 0

        pbar = build_pbar(self.n_samples)

        self.config.logger('\n%s\n' % name)

        g = generator.generate(exhaustive=exhaustive)

        while True:
            pbar.update(counter)
            # Each call to next results in a batch of possible
            # corrections, only one of which is correct.
            try:
                next_batch = next(g)
            except StopIteration:
                break

            if isinstance(next_batch, (tuple, list)):
                d, sample_weight = next_batch
            else:
                assert isinstance(next_batch, dict)
                d = next_batch
                sample_weight = None

            targets = d[self.config.target_name]
            pred = self.model.predict(d, verbose=0)[self.config.target_name]

            y.extend(targets[:, 1].tolist())

            y_hat_tmp = [0] * len(targets)
            y_hat_tmp[np.argmax(pred[:, 1])] = 1
            y_hat.extend(y_hat_tmp)
            if targets[:, 1][np.argmax(pred[:, 1])] == 1:
                y_hat_binary.append(1)
            else:
                y_hat_binary.append(0)

            correct_word = d['correct_word'][0]

            y_hat_dictionary_tmp = []
            if d['candidate_word'][0] == correct_word:
                y_hat_dictionary_binary.append(1)
            else:
                y_hat_dictionary_binary.append(0)

            for i,c in enumerate(d['candidate_word']):
                # The first word in the results returned by the dictionary
                # is the dictionary's highest-scoring candidate for
                # replacing the non-word.
                if i == 0:
                    y_hat_dictionary_tmp.append(1)
                else:
                    y_hat_dictionary_tmp.append(0)
            y_hat_dictionary.extend(y_hat_dictionary_tmp)

            if len(y_hat_dictionary_tmp) != len(targets):
                raise ValueError('non_word %s correct_word %s dictlen %d targetslen %d' %
                        (d['non_word'][0], d['correct_word'][0],
                            len(y_hat_dictionary_tmp),
                            len(targets)))

            counter += 1
            if counter >= self.n_samples:
                break

        pbar.finish()

        self.config.logger('\n')
        self.config.logger('Dictionary %s binary accuracy %.04f accuracy %.04f F1 %0.4f' % 
                (
                    name,
                    sum(y_hat_dictionary_binary) / float(len(y_hat_dictionary_binary)),
                    accuracy_score(y, y_hat_dictionary),
                    f1_score(y, y_hat_dictionary)
                ))
        self.config.logger('Dictionary confusion matrix')
        self.config.logger(confusion_matrix(y, y_hat_dictionary))

        model_binary_accuracy = sum(y_hat_binary) / float(len(y_hat_binary))
        model_accuracy = accuracy_score(y, y_hat)
        model_f1 = f1_score(y, y_hat)

        self.config.logger('\n')
        self.config.logger('ConvNet %s binary accuracy %.04f accuracy %.04f F1 %0.4f' % 
                (name, model_binary_accuracy, model_accuracy, model_f1))
        self.config.logger('ConvNet confusion matrix')
        self.config.logger(confusion_matrix(y, y_hat))

        self.config.logger('\n')

        if do_callbacks:
            logs['f1'] = model_f1
            for cb in self.callbacks:
                cb.on_epoch_end(epoch, logs)


    def on_epoch_end(self, epoch, logs={}):
        self.compute_metrics(self.generator, 'validation',
                self.config.fixed_callback_validation_data,
                epoch, logs, do_callbacks=True)

        for name,(gen,exhaustive) in self.other_generators.items():
            self.compute_metrics(gen, name, exhaustive,
                    epoch, logs, do_callbacks=False)

def build_callbacks(config, generator, n_samples, other_generators={}):
    # For this model, we want to monitor F1 for early stopping and
    # model checkpointing.  The way to do that is for the metrics callback
    # compute F1, put it in the logs dictionary that's passed to
    # on_epoch_end, and to pass that to the early stopping and model
    # checkpointing callbacks.
    controller_callbacks = []
    controller_callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor=config.callback_monitor,
                mode=config.callback_monitor_mode,
                patience=config.patience,
                verbose=1))

    if 'persistent' in config.mode:
        controller_callbacks.append(
                keras.callbacks.ModelCheckpoint(
                    monitor=config.callback_monitor,
                    mode=config.callback_monitor_mode,
                    filepath=config.model_path + config.model_checkpoint_file,
                    save_best_only=config.save_best_only,
                    verbose=1))

    controller = MetricsCallback(config, generator, n_samples,
            callbacks=controller_callbacks,
            other_generators=other_generators)

    callbacks = []
    callbacks.append(controller)
    callbacks.append(modeling.callbacks.DenseWeightNormCallback(config))

    return callbacks

retriever_lock = threading.Lock()

# Retrievers
def build_retriever(vocabulary, n_random_candidates=0, max_candidates=0, n_neighbors=0, bottom_candidates=0, exclude_candidates_containing_hyphen_or_space=False, only_return_candidates_in_vocabulary=False):
    with retriever_lock:
        retrievers = []
        retrievers.append(spelldict.AspellRetriever())
        retrievers.append(spelldict.EditDistanceRetriever(vocabulary))
        if n_random_candidates > 0:
            retrievers.append(spelldict.RandomRetriever(
                vocabulary, n_random_candidates))
        if n_neighbors > 0:
            estimator = NearestNeighbors(metric='hamming', algorithm='auto')
            retrievers.append(
                    spelldict.NearestNeighborsRetriever(
                        vocabulary, estimator))

        retriever = spelldict.RetrieverCollection(retrievers)
        jaro_sorter = spelldict.DistanceSorter('jaro_winkler')
        retriever = spelldict.SortingRetriever(retriever, jaro_sorter)
        retriever = spelldict.CachingRetriever(retriever,
                cache_dir='/localwork/ndronen/spelling/spelling_error_cache/multiclass/')

        if exclude_candidates_containing_hyphen_or_space:
            fltr = lambda word: ' ' not in word and '-' not in word
            retriever = spelldict.FilteringRetriever(retriever, fltr)

        if only_return_candidates_in_vocabulary:
            filter_vocabulary = set(vocabulary)
            fltr = lambda word: word in filter_vocabulary
            retriever = spelldict.FilteringRetriever(retriever, fltr)

        if max_candidates > 0:
            retriever = spelldict.TopKRetriever(retriever, max_candidates)
        elif bottom_candidates > 0:
            retriever = spelldict.BottomKRetriever(retriever, bottom_candidates)
        return retriever

def count_contexts(contexts):
    return sum([len(c) for c in contexts.values()])

def build_vocabulary(word_to_context, config):
    # Fit the vocabulary over all the contexts.  For a word in any
    # context to be included in the vocabulary, it must occur at least
    # `min_df` times in the corpus of contexts.
    all_contexts = []
    for word,contexts in word_to_context.items():
        all_contexts.append(word)
        # De-tokenize just for the CountVectorizer.
        all_contexts.extend([' '.join(c) for c in contexts])
    vectorizer = CountVectorizer(min_df=config.count_vectorizer_min_df)
    vectorizer.fit(all_contexts)

    vocabulary = vectorizer.vocabulary_

    # TODO: figure out why we're adding '^' and '$' to the vocabulary,
    # and ASCII letters.
    vocabulary[MulticlassContextTransformer.UNKNOWN_WORD] = len(vocabulary)
    vocabulary[MulticlassContextTransformer.NON_WORD] = len(vocabulary)
    # Start-of-word marker.
    vocabulary['^'] = len(vocabulary)
    # End-of-word marker.
    vocabulary['$'] = len(vocabulary)
    # ASCII letters.
    for letter in string.ascii_letters:
        if letter not in vocabulary:
            vocabulary[letter] = len(vocabulary)

    df = pd.read_csv('~/proj/spelling/data/aspell-dict.csv.gz', sep='\t', encoding='utf8')
    for word in df.word.tolist():
        if word not in vocabulary:
            vocabulary[word] = len(vocabulary)

    return vocabulary

def load_data(config):
    loader = MulticlassLoader(pickle_path=config.pickle_path)
    word_to_context = loader.load()

    config.logger('%d contexts before filtering' %
            count_contexts(word_to_context))

    # Drop any contexts that contain 'digit'.
    filter = DigitFilter()
    for word,contexts in word_to_context.items():
        word_to_context[word] = filter.transform(contexts)

    # Filter by length of word.
    config.logger('%d word_to_context words before length filtering' %
            len(word_to_context))

    for word in list(word_to_context.keys()):
        if len(word) < config.min_length or len(word) > config.max_length:
            del word_to_context[word]

    config.logger('%d word_to_context words after length filtering' %
            len(word_to_context))

    # Filter by number of contexts in which word appears.
    context_lengths = []
    words_to_delete = []
    for word,contexts in word_to_context.items():
        if len(contexts) < config.min_contexts or len(contexts) > config.max_contexts:
            words_to_delete.append(word)
            continue
        context_lengths.append(len(contexts))

    for word in words_to_delete:
        del word_to_context[word]

    config.logger('%d word_to_context words after context length filtering' %
            len(word_to_context))

    config.logger('min contexts %d max %d' %
            (min(context_lengths), max(context_lengths)))

    config.logger('%d contexts after filtering' %
            count_contexts(word_to_context))

    # Tokenize the contexts.
    tokenizer = ContextTokenizer()
    for word,contexts in word_to_context.items():
        word_to_context[word] = tokenizer.transform(contexts)

    # Turn the contexts into just their windows.
    windower = ContextWindowTransformer(window_size=config.context_input_width)
    for word,contexts in word_to_context.items():
        word_to_context[word] = windower.transform(contexts, [word] * len(contexts))

    return word_to_context

def setup(config):
    word_to_context = load_data(config)
    config.logger('%d keys in word_to_context' % len(word_to_context))
    vocabulary = build_vocabulary(word_to_context, config)

    lengths = [len(word) for word in word_to_context.keys()]
    config.logger('vocabulary min length %d max length %d' %
            (min(lengths), max(lengths)))

    config.n_context_embeddings = len(vocabulary)
    config.logger('n_context_embeddings %d' % config.n_context_embeddings)

    n_classes = 2

    data = []
    for word,v in word_to_context.items():
        for context in v:
            data.append((word,context))

    splitter = globals()[config.dataset_splitter](
            word_to_context,
            train_size=config.train_size,
            validation_size=config.validation_size)

    train_data, valid_data, test_data = splitter.split()

    config.logger('train %d validation %d test %d' % (
        len(train_data),
        len(valid_data),
        len(test_data)))

    non_word_generator = globals()[config.non_word_generator](
            min_edits=config.min_edits,
            max_edits=config.max_edits,
            vocabulary=vocabulary)

    char_input_transformer = MulticlassNonwordTransformer(
            output_width=config.char_input_width)

    real_word_input_transformer = MulticlassContextTransformer(
            vocabulary=vocabulary,
            output_width=1)

    context_input_transformer = MulticlassContextTransformer(
            vocabulary=vocabulary,
            output_width=config.context_input_width)

    train_retriever_vocabulary = vocabulary
    if config.train_only_return_candidates_in_vocabulary:
        train_retriever_vocabulary = list(word_to_context.keys())

    train_retriever = build_retriever(
                train_retriever_vocabulary,
                n_random_candidates=config.n_random_train_candidates,
                max_candidates=config.max_train_candidates,
                n_neighbors=config.n_train_neighbors,
                bottom_candidates=config.bottom_k_candidates_only,
                exclude_candidates_containing_hyphen_or_space=config.exclude_candidates_containing_hyphen_or_space,
                only_return_candidates_in_vocabulary=config.train_only_return_candidates_in_vocabulary)

    config.logger('train_retriever %s' % str(type(train_retriever)))

    # To be fair to the dictionary baseline, the retriever used for
    # validation should only consider candidates from the set of words
    # that the model is trained to correct.  We do it this way because
    # when the model is trained to correct a small subset of a dictionary,
    # such as when focusing on short words, the model might learn that
    # the correct word for the training examples belongs to the small
    # subset; if we don't remove from the dictionary the words that are
    # not in that subset, the candidate list returned by the dictionary
    # may contain words that are never the true correction, and the model
    # may learn to detect this.
    valid_retriever_vocabulary = vocabulary
    if config.valid_only_return_candidates_in_vocabulary:
        valid_retriever_vocabulary = list(word_to_context.keys())
    valid_retriever = build_retriever(
                valid_retriever_vocabulary,
                exclude_candidates_containing_hyphen_or_space=config.exclude_candidates_containing_hyphen_or_space,
                only_return_candidates_in_vocabulary=config.valid_only_return_candidates_in_vocabulary)

    if config.preload_retriever_cache:
        train_retriever.load_cache(verbose=True)
        valid_retriever.load_cache(verbose=True)

    train_generator = BinaryGenerator(train_data,
            non_word_generator,
            char_input_transformer,
            real_word_input_transformer,
            context_input_transformer,
            non_word_char_input_name=config.non_word_char_input_name,
            real_word_char_input_name=config.real_word_char_input_name,
            real_word_input_name=config.real_word_input_name,
            context_input_name=config.context_input_name,
            target_name=config.target_name,
            retriever=train_retriever,
            n_classes=n_classes,
            sample_weight_exponent=config.class_weight_exponent,
            batch_size=config.batch_size,
            use_real_word_examples=config.use_real_word_examples)

    config.logger('a validation set example' + str(valid_data[0]))

    valid_generator = BinaryGenerator(valid_data,
            non_word_generator,
            char_input_transformer,
            real_word_input_transformer,
            context_input_transformer,
            non_word_char_input_name=config.non_word_char_input_name,
            real_word_char_input_name=config.real_word_char_input_name,
            real_word_input_name=config.real_word_input_name,
            context_input_name=config.context_input_name,
            target_name=config.target_name,
            retriever=valid_retriever,
            n_classes=n_classes,
            sample_weight_exponent=config.class_weight_exponent,
            n_samples=config.n_val_samples)

    test_generator = BinaryGenerator(test_data,
            non_word_generator,
            char_input_transformer,
            real_word_input_transformer,
            context_input_transformer,
            non_word_char_input_name=config.non_word_char_input_name,
            real_word_char_input_name=config.real_word_char_input_name,
            real_word_input_name=config.real_word_input_name,
            context_input_name=config.context_input_name,
            target_name=config.target_name,
            retriever=build_retriever(list(vocabulary.keys())),
            n_classes=n_classes,
            sample_weight_exponent=config.class_weight_exponent)

    return {
            'data': data,
            'n_classes': n_classes,
            'word_to_context': word_to_context,
            'vocabulary': vocabulary,

            'non_word_generator': non_word_generator,
            'char_input_transformer': char_input_transformer,
            'real_word_input_transformer': real_word_input_transformer,
            'context_input_transformer': context_input_transformer,

            'train_generator': train_generator,
            'train_data': train_data,
            'train_retreiver': train_retriever,

            'valid_generator': valid_generator,
            'valid_data': valid_data,
            'valid_retreiver': valid_retriever,

            'test_generator': test_generator,
            'test_data': test_data,
            }


def fit(config):
    setup_vars = setup(config)

    context_embedding_weights = None

    if config.use_context_model or config.use_real_word_embedding:
        try:
            vectors_path = config.vectors_path

            with gzip.open(vectors_path, 'rb') as f:
                vectors_dict = pickle.load(f, encoding='latin1')

            embedding_shape = (config.n_context_embeddings,
                        config.n_pretrained_context_embed_dims)
            config.logger("embedding_shape " + str(embedding_shape))
            context_embedding_weights = np.random.uniform(
                    -0.05, 0.05, size=embedding_shape)

            n_pretrained = 0

            for word in setup_vars['vocabulary']:
                if word in vectors_dict:
                    row = setup_vars['vocabulary'][word]
                    context_embedding_weights[row] = vectors_dict[word]
                    n_pretrained += 1

            config.logger('using %d pretrained word vectors out of %d' %
                    (n_pretrained, len(setup_vars['vocabulary'])))

            if config.n_context_embed_dims < config.n_pretrained_context_embed_dims:
                # Reduce the dimensionality of the pretrained vectors.
                pca = PCA(n_components=config.n_context_embed_dims)
                context_embedding_weights = pca.fit_transform(
                        context_embedding_weights)
        except (AttributeError, TypeError):
            pass

    graph = build_model(config, setup_vars['n_classes'],
            context_embedding_weights=context_embedding_weights)

    config.logger('model has %d parameters' % graph.count_params())

    config.logger('n_val_samples %d' % config.n_val_samples)
    config.logger('building callbacks')

    callbacks = build_callbacks(config,
            setup_vars['valid_generator'],
            n_samples=config.n_val_samples)

    scheduler = Schedule(graph, config.learning_rate_schedule)
    callbacks.append(
            keras.callbacks.LearningRateScheduler(scheduler.schedule))

    # Don't use class weights here; the targets are balanced.
    class_weight = {}

    verbose = 2 if 'background' in config.mode else 1

    #print(next(setup_vars['train_generator'].generate(train=True)))
    #print(next(setup_vars['valid_generator'].generate(exhaustive=True)))

    if 'background' in config.mode:
        verbose = 2
        with open(config.model_dest + '/model.yaml', 'w') as f:
            f.write(graph.to_yaml())
    else:
        verbose = 1

    # TODO: make validation generator run the same, fixed subset
    # of the data at the end of every epoch.  
    graph.fit_generator(setup_vars['train_generator'].generate(exhaustive=False),
            samples_per_epoch=config.samples_per_epoch,
            nb_worker=config.n_worker,
            nb_epoch=config.n_epoch,
            validation_data=setup_vars['valid_generator'].generate(exhaustive=True),
            nb_val_samples=config.n_val_samples,
            callbacks=callbacks,
            class_weight=class_weight,
            verbose=verbose)
