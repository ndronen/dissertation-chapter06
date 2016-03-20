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
        LearnedEditTransformer)

from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.utils import check_random_state

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
from keras.layers.core import Dense, Dropout, Activation, Flatten, Layer, Lambda, Reshape
from keras.constraints import maxnorm
from keras.layers.noise import GaussianNoise
from keras.layers.normalization import BatchNormalization
import keras.callbacks

import modeling.data
from modeling.builders import (build_embedding_layer,
    build_convolutional_layer, build_pooling_layer,
    build_dense_layer, build_optimizer, load_weights,
    build_hierarchical_softmax_layer)
from modeling.utils import balanced_class_weights

import spelling.dictionary as spelldict

class Identity(Layer):
    def get_output(self, train):
        return self.get_input(train)

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

def build_model(config, n_classes):
    np.random.seed(config.random_state)

    graph = Graph()

    #######################################################################
    # Character layers
    #######################################################################

    # Character-level input for the non-word error.
    graph.add_input(config.non_word_char_input_name,
            input_shape=(config.char_input_width,), dtype='int')
    # Character-level input for the real word suggestion.
    graph.add_input(config.real_word_char_input_name,
            input_shape=(config.char_input_width,), dtype='int')

    char_model = Sequential()
    char_model.add(build_embedding_layer(config,
            input_width=config.char_input_width,
            n_embeddings=config.n_char_embeddings,
            n_embed_dims=config.n_char_embed_dims))
    char_model.add(build_convolutional_layer(config,
            n_filters=config.n_char_filters,
            filter_width=config.char_filter_width))
    if config.batch_normalization:
        char_model.add(BatchNormalization())
    char_model.add(Activation('relu'))
    char_model.add(build_pooling_layer(config,
            input_width=config.char_input_width,
            filter_width=config.char_filter_width))
    char_model.add(Flatten())

    graph.add_shared_node(char_model,
            name='char_model',
            inputs=[
                config.non_word_char_input_name,
                config.real_word_char_input_name],
            outputs=[
                'non_word_char_model',
                'real_word_char_model'])

    if config.char_merge_mode == 'cos':
        char_dot_axes = ([1], [1])
    else:
        char_dot_axes = -1

    graph.add_node(Dense(config.char_merge_n_hidden),
            name='char_merge_dense',
        inputs=['non_word_char_model', 'real_word_char_model'],
        merge_mode=config.char_merge_mode,
        dot_axes=char_dot_axes)
    if config.char_merge_act == "sigmoid":
        lambda_layer = Lambda(lambda x: 6.0*x)
    elif config.char_merge_act == "tanh":
        lambda_layer = Lambda(lambda x: 3.0*x)
    else:
        lambda_layer = Lambda(lambda x: x)
    graph.add_node(lambda_layer,
            name='char_merge_scale',
            input='char_merge_dense')
    graph.add_node(Activation(config.char_merge_act),
            name='char_merge',
            input='char_merge_scale')

    #######################################################################
    # Word layers
    #######################################################################

    # Word-level input for the context of the non-word error.
    graph.add_input(config.context_input_name,
            input_shape=(config.context_input_width,), dtype='int')

    # Context embedding.
    context_embedding = build_embedding_layer(config,
            input_width=config.context_input_width,
            n_embeddings=config.n_context_embeddings,
            n_embed_dims=config.n_context_embed_dims)

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

    if config.merge_mode == 'cos':
        dot_axes = ([1], [1])
    else:
        dot_axes = -1

    # Add some number of fully-connected layers without skip connections.
    prev_layer = None
    for i,n_hidden in enumerate(config.fully_connected):
        layer_name = 'dense%02d' %i
        l = build_dense_layer(config, n_hidden=n_hidden)
        if i == 0:
            inputs = ['context_flatten', 'non_word_char_model']
            if config.use_real_word_embedding:
                inputs.append('real_word_reshape')
            graph.add_node(l, name=layer_name,
                    inputs=inputs,
                    merge_mode=config.merge_mode,
                    dot_axes=dot_axes)
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
                    graph.add_node(Dropout(config.dropout_fc_p), name=layer_name+'do', input=prev_layer)
                    prev_layer = layer_name+'do'

        graph.add_node(Identity(), name=block_name+'output', inputs=[block_input_layer, prev_layer], merge_mode='sum')
        graph.add_node(Activation('relu'), name=block_name+'relu', input=block_name+'output')
        prev_layer = block_input_layer = block_name+'relu'

    softmax_inputs = ['char_merge']
    if prev_layer is None:
        softmax_inputs.append('context_flatten')
        if config.use_real_word_embedding:
            softmax_inputs.append('real_word_reshape')
        graph.add_node(Dense(n_classes, init=config.dense_init,
            W_constraint=maxnorm(config.softmax_max_norm)),
            name='softmax',
            inputs=softmax_inputs,
            merge_mode=config.merge_mode,
            dot_axes=dot_axes)
    else:
        softmax_inputs.append(prev_layer)
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
def build_retriever(vocabulary):
    with retriever_lock:
        aspell_retriever = spelldict.AspellRetriever()
        edit_distance_retriever = spelldict.EditDistanceRetriever(vocabulary)
        retriever = spelldict.RetrieverCollection([aspell_retriever, edit_distance_retriever])
        retriever = spelldict.CachingRetriever(retriever,
                cache_dir='/localwork/ndronen/spelling/spelling_error_cache/')
        jaro_sorter = spelldict.DistanceSorter('jaro_winkler')
        return spelldict.SortingRetriever(retriever, jaro_sorter)

def fit(config, callbacks=[]):
    loader = MulticlassLoader(pickle_path=config.pickle_path)
    word_to_context = loader.load()

    # Tokenize the contexts.
    tokenizer = ContextTokenizer()
    for word,contexts in word_to_context.items():
        word_to_context[word] = tokenizer.transform(contexts)

    # Turn the contexts into just their windows.
    windower = ContextWindowTransformer(window_size=config.context_input_width)
    for word,contexts in word_to_context.items():
        word_to_context[word] = windower.transform(contexts, [word] * len(contexts))

    # Fit the vocabulary over all the contexts.
    all_contexts = []
    for word,contexts in word_to_context.items():
        all_contexts.append(word)
        # De-tokenize just for the CountVectorizer.
        all_contexts.extend([' '.join(c) for c in contexts])
    vectorizer = CountVectorizer()
    vectorizer.fit(all_contexts)

    vocabulary = vectorizer.vocabulary_
    vocabulary[''] = len(vocabulary)
    vocabulary['<UNK>'] = len(vocabulary)
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
            retriever=build_retriever(vocabulary.keys()),
            n_classes=n_classes)

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
            retriever=build_retriever(vocabulary.keys()),
            n_classes=n_classes)

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
            retriever=build_retriever(vocabulary.keys()),
            n_classes=n_classes)

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

    graph.fit_generator(train_generator.generate(train=True),
            samples_per_epoch=config.samples_per_epoch,
            nb_worker=config.n_worker,
            nb_epoch=config.n_epoch,
            validation_data=valid_generator.generate(exhaustive=True),
            nb_val_samples=config.n_val_samples,
            callbacks=callbacks,
            class_weight=class_weight,
            verbose=verbose)