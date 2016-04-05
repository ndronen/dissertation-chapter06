import re
import string
import random

from modeling.callbacks import DenseWeightNormCallback
from chapter06.dataset import (
        MulticlassLoader,
        MulticlassSplitter,
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
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.utils import check_random_state
from sklearn.neighbors import NearestNeighbors
from keras.constraints import maxnorm

from spelling.edits import Editor
from spelling.utils import build_progressbar

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
    graph.add_node(Activation('relu'), name=relu_name, input=prev_layer)
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

    graph.add_shared_node(Activation('relu'),
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

    graph.add_shared_node(Flatten(),
            name='char_flatten',
            inputs=['non_word_conv_pool',
                'real_word_conv_pool'],
            outputs=['non_word_flatten',
                'real_word_flatten'])

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
            inputs=['non_word_flatten',
                'real_word_flatten'],
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
    char_merge_output = 'char_merge_act'

    graph.add_node(Activation(config.char_merge_act),
            name=char_merge_output,
            input=prev_char_layer)

    return non_word_output, char_merge_output

def build_context_model(graph, config):
    # Word-level input for the context of the non-word error.
    graph.add_input(config.context_input_name,
            input_shape=(config.context_input_width,), dtype='int')

    # Context embedding.
    context_embedding = build_embedding_layer(config,
            input_width=config.context_input_width,
            n_embeddings=config.n_context_embeddings,
            n_embed_dims=config.n_context_embed_dims,
            dropout=config.dropout_embedding_p)

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

    return real_word_output, context_output

def build_model(config, n_classes):
    np.random.seed(config.random_state)

    graph = Graph()

    #######################################################################
    # Character layers
    #######################################################################

    dense_inputs = []
    softmax_inputs = []

    if config.use_char_model:
        non_word_output, char_merge_output = build_char_model(graph, config)

        if config.non_word_gaussian_noise_sd > 0.:
            graph.add_node(GaussianNoise(config.non_word_gaussian_noise_sd),
                    input=non_word_output,
                    name='non_word_output_noise')
            non_word_output = 'non_word_output_noise'

        dense_inputs.append(non_word_output)
        softmax_inputs.append(char_merge_output)

    #######################################################################
    # Word layers
    #######################################################################

    if config.use_context_model:
        real_word_output, context_output = build_context_model(graph, config)
        dense_inputs.append(context_output)
        if config.use_real_word_embedding:
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
                        input=dense_inputs[0],
                        merge_mode=config.merge_mode,
                        dot_axes=dot_axes)
            else:
                graph.add_node(l, name=layer_name,
                        inputs=dense_inputs,
                        merge_mode=config.merge_mode,
                        dot_axes=dot_axes)
            if config.dense_gaussian_noise_sd > 0.:
                graph.add_node(
                        GaussianNoise(config.dense_gaussian_noise_sd),
                        name=layer_name+'gaussian',
                        input=layer_name)
            prev_layer = layer_name+'gaussian'
        else:
            graph.add_node(l, name=layer_name, input=prev_layer)
        prev_layer = layer_name
        if config.batch_normalization:
            graph.add_node(BatchNormalization(), name=layer_name+'bn', input=prev_layer)
            prev_layer = layer_name+'bn'
        graph.add_node(Activation('relu'), name=layer_name+'relu', input=prev_layer)
        prev_layer = layer_name+'relu'
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
                    W_constraint=maxnorm(config.residual_max_norm))
            graph.add_node(l, name=layer_name, input=prev_layer)
            prev_layer = layer_name
    
            if config.batch_normalization:
                graph.add_node(BatchNormalization(), name=layer_name+'bn', input=prev_layer)
                prev_layer = layer_name+'bn'
    
            if i < n_layers_per_residual_block:
                graph.add_node(Activation('relu'), name=layer_name+'relu', input=prev_layer)
                prev_layer = layer_name+'relu'
                if config.dropout_fc_p > 0.:
                    graph.add_node(Dropout(config.dropout_residual_p), name=layer_name+'do', input=prev_layer)
                    prev_layer = layer_name+'do'

        graph.add_node(Identity(), name=block_name+'output', inputs=[block_input_layer, prev_layer], merge_mode='sum')
        graph.add_node(Activation('relu'), name=block_name+'relu', input=block_name+'output')
        prev_layer = block_input_layer = block_name+'relu'

    if prev_layer is None:
        softmax_inputs.extend(dense_inputs)
        graph.add_node(Dense(n_classes, init=config.dense_init,
            W_constraint=maxnorm(config.softmax_max_norm)),
            name='softmax',
            inputs=softmax_inputs,
            merge_mode=config.merge_mode,
            dot_axes=dot_axes)
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
        pbar = build_progressbar(self.n_samples)
        print('\n%s\n' % name)

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
def build_retriever(vocabulary, n_random_candidates=0, max_candidates=0, n_neighbors=0):
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
        if max_candidates > 0:
            retriever = spelldict.TopKRetriever(retriever, max_candidates)
        return retriever

def count_contexts(contexts):
    return sum([len(c) for c in contexts.values()])

def fit(config, callbacks=[]):
    loader = MulticlassLoader(pickle_path=config.pickle_path)
    word_to_context = loader.load()

    config.logger('%d contexts before digit filtering' %
            count_contexts(word_to_context))

    # Drop any contexts that contain 'digit'.
    # TODO: possibly eliminate this step; it may not be necessary now
    # that we're using a non-default min_df argument to CountVectorizer.
    filter = DigitFilter()
    for word,contexts in word_to_context.items():
        word_to_context[word] = filter.transform(contexts)

    config.logger('%d contexts after digit filtering' %
            count_contexts(word_to_context))

    # Tokenize the contexts.
    tokenizer = ContextTokenizer()
    for word,contexts in word_to_context.items():
        word_to_context[word] = tokenizer.transform(contexts)

    # Turn the contexts into just their windows.
    windower = ContextWindowTransformer(window_size=config.context_input_width)
    for word,contexts in word_to_context.items():
        word_to_context[word] = windower.transform(contexts, [word] * len(contexts))

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

    # TODO: figure out why we're adding '^' and '$' to the vocabulary.
    vocabulary = vectorizer.vocabulary_
    vocabulary[MulticlassContextTransformer.UNKNOWN_WORD] = len(vocabulary)
    vocabulary[MulticlassContextTransformer.NON_WORD] = len(vocabulary)
    # Start-of-word marker.
    vocabulary['^'] = len(vocabulary)
    # End-of-word marker.
    vocabulary['$'] = len(vocabulary)
    for letter in string.ascii_letters:
        if letter not in vocabulary:
            vocabulary[letter] = len(vocabulary)

    df = pd.read_csv('~/proj/spelling/data/aspell-dict.csv.gz', sep='\t', encoding='utf8')
    for word in df.word.tolist():
        if word not in vocabulary:
            vocabulary[word] = len(vocabulary)

    config.n_context_embeddings = len(vocabulary)
    print('n_context_embeddings %d' % config.n_context_embeddings)

    n_classes = 2

    splitter = MulticlassSplitter(word_to_context=word_to_context)
    train_data, valid_data, test_data = splitter.split()

    def dataset_size(dataset):
        return sum([len(v) for v in dataset.values()])

    print('train %d validation %d test %d' % (
        dataset_size(train_data),
        dataset_size(valid_data),
        dataset_size(test_data)))

    non_word_generator = globals()[config.non_word_generator](
            min_edits=config.min_edits,
            max_edits=config.max_edits)

    char_input_transformer = MulticlassNonwordTransformer(
            output_width=config.char_input_width)

    real_word_input_transformer = MulticlassContextTransformer(
            vocabulary=vocabulary,
            output_width=1)

    context_input_transformer = MulticlassContextTransformer(
            vocabulary=vocabulary,
            output_width=config.context_input_width)

    train_retriever = build_retriever(
                list(vocabulary.keys()), 
                n_random_candidates=config.n_random_train_candidates,
                max_candidates=config.max_train_candidates,
                n_neighbors=config.n_train_neighbors)

    valid_retriever = build_retriever(list(vocabulary.keys()))

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
            sample_weight_exponent=config.class_weight_exponent)

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
            sample_weight_exponent=config.class_weight_exponent)

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

    graph = build_model(config, n_classes)

    config.logger('model has %d parameters' % graph.count_params())

    config.logger('building callbacks')
    callbacks = build_callbacks(config,
            valid_generator,
            n_samples=config.n_val_samples)

    # Don't use class weights here; the targets are balanced.
    class_weight = {}

    verbose = 2 if 'background' in config.mode else 1

    #print(next(train_generator.generate(train=True)))
    #print(next(valid_generator.generate(exhaustive=True)))

    if 'background' in config.mode:
        verbose = 2
        with open(config.model_dest + '/model.yaml', 'w') as f:
            f.write(graph.to_yaml())
    else:
        verbose = 1

    graph.fit_generator(train_generator.generate(train=True),
            samples_per_epoch=config.samples_per_epoch,
            nb_worker=config.n_worker,
            nb_epoch=config.n_epoch,
            validation_data=valid_generator.generate(exhaustive=True),
            nb_val_samples=config.n_val_samples,
            callbacks=callbacks,
            class_weight=class_weight,
            verbose=verbose)

#def build_flat_context_model(config, n_classes):
#    np.random.seed(config.random_state)
#
#    graph = Graph()
#
#    #######################################################################
#    # Character layers
#    #######################################################################
#    non_word_output, char_merge_output = build_char_model(graph, config)
#
#    #######################################################################
#    # Word layers
#    #######################################################################
#
#    # Word-level input for the context of the non-word error.
#    context_embedding_inputs = []
#    context_embedding_outputs = []
#    context_reshape_outputs = []
#    for i in range(1, 6):
#        name = '%s_%02d' % (config.context_input_name, i)
#        graph.add_input(name, input_shape=(1,), dtype='int')
#        context_embedding_inputs.append(name)
#        context_embedding_outputs.append(
#                'context_embedding_%02d' % i)
#        context_reshape_outputs.append(
#                'context_reshape_%02d' % i)
#
#    context_embedding = build_embedding_layer(config,
#            input_width=1,
#            n_embeddings=config.n_context_embeddings,
#            n_embed_dims=config.n_context_embed_dims,
#            dropout=config.dropout_embedding_p)
#
#    graph.add_shared_node(context_embedding,
#            name='context_embedding',
#            inputs=context_embedding_inputs,
#            outputs=context_embedding_outputs)
#
#    graph.add_shared_node(Reshape((config.n_context_embed_dims,)),
#        name='context_reshape',
#        inputs=context_embedding_outputs,
#        outputs=context_reshape_outputs)
#
#    graph.add_node(Identity(),
#            name='word_context',
#            inputs=context_reshape_outputs,
#            merge_mode='concat')
#
#    # Add some number of fully-connected layers without skip connections.
#    prev_layer = None
#    for i,n_hidden in enumerate(config.fully_connected):
#        layer_name = 'dense%02d' %i
#        l = build_dense_layer(config, n_hidden=n_hidden,
#                max_norm=config.dense_max_norm)
#        if i == 0:
#            inputs = ['word_context', non_word_output]
#            graph.add_node(l, name=layer_name,
#                    inputs=inputs,
#                    merge_mode='concat')
#        else:
#            graph.add_node(l, name=layer_name, input=prev_layer)
#        prev_layer = layer_name
#        if config.batch_normalization:
#            graph.add_node(BatchNormalization(), name=layer_name+'bn', input=prev_layer)
#            prev_layer = layer_name+'bn'
#        graph.add_node(Activation('relu'), name=layer_name+'relu', input=prev_layer)
#        prev_layer = layer_name+'relu'
#        if config.dropout_fc_p > 0.:
#            graph.add_node(Dropout(config.dropout_fc_p), name=layer_name+'do', input=prev_layer)
#            prev_layer = layer_name+'do'
#
#    # Add sequence of residual blocks.
#    for i in range(config.n_residual_blocks):
#        # Add a fixed number of layers per residual block.
#        block_name = '%02d' % i
#
#        graph.add_node(Identity(), name=block_name+'input', input=prev_layer)
#        prev_layer = block_input_layer = block_name+'input'
#
#        try:
#            n_layers_per_residual_block = config.n_layers_per_residual_block
#        except AttributeError:
#            n_layers_per_residual_block = 2
#
#        for layer_num in range(n_layers_per_residual_block):
#            layer_name = 'h%s%02d' % (block_name, layer_num)
#    
#            l = Dense(config.n_hidden_residual, init=config.residual_init,
#                    W_constraint=maxnorm(config.residual_max_norm))
#            graph.add_node(l, name=layer_name, input=prev_layer)
#            prev_layer = layer_name
#    
#            if config.batch_normalization:
#                graph.add_node(BatchNormalization(), name=layer_name+'bn', input=prev_layer)
#                prev_layer = layer_name+'bn'
#    
#            if i < n_layers_per_residual_block:
#                graph.add_node(Activation('relu'), name=layer_name+'relu', input=prev_layer)
#                prev_layer = layer_name+'relu'
#                if config.dropout_fc_p > 0.:
#                    graph.add_node(Dropout(config.dropout_residual_p), name=layer_name+'do', input=prev_layer)
#                    prev_layer = layer_name+'do'
#
#        graph.add_node(Identity(), name=block_name+'output', inputs=[block_input_layer, prev_layer], merge_mode='sum')
#        graph.add_node(Activation('relu'), name=block_name+'relu', input=block_name+'output')
#        prev_layer = block_input_layer = block_name+'relu'
#
#    softmax_inputs = [char_merge_output]
#    if prev_layer is None:
#        softmax_inputs.append('word_context')
#        # Not ready to do this yet -- it used to only happen at the
#        # beginning of the fully-connected block.
#        #softmax_inputs.append(non_word_output)
#        graph.add_node(Dense(n_classes, init=config.dense_init,
#            W_constraint=maxnorm(config.softmax_max_norm)),
#            name='softmax',
#            inputs=softmax_inputs,
#            merge_mode=config.merge_mode,
#            dot_axes=dot_axes)
#    else:
#        softmax_inputs.append(prev_layer)
#        graph.add_node(Dense(n_classes, init=config.dense_init,
#            W_constraint=maxnorm(config.softmax_max_norm)),
#            name='softmax',
#            inputs=softmax_inputs)
#    prev_layer = 'softmax'
#    if config.batch_normalization:
#        graph.add_node(BatchNormalization(), name='softmax_bn', input='softmax')
#        prev_layer = 'softmax_bn'
#    graph.add_node(Activation('softmax'), name='softmax_activation', input=prev_layer)
#    graph.add_output(name='binary_correction_target', input='softmax_activation')
#
#    load_weights(config, graph)
#
#    optimizer = build_optimizer(config)
#
#    graph.compile(loss={'binary_correction_target': config.loss}, optimizer=optimizer)
#
#    return graph
