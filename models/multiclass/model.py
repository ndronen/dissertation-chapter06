import string

from modeling.callbacks import DenseWeightNormCallback
from chapter06.dataset import (
        MulticlassLoader,
        MulticlassSplitter,
        MulticlassGenerator,
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
from keras.layers.core import Dense, Dropout, Activation, Flatten, Layer
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

    # Character-level input for the non-word error.
    graph.add_input(config.non_word_char_input_name,
            input_shape=(config.non_word_char_input_width,), dtype='int')
    non_word_embedding = build_embedding_layer(config,
            input_width=config.non_word_char_input_width,
            n_embeddings=config.n_char_embeddings,
            n_embed_dims=config.n_char_embed_dims)
    graph.add_node(non_word_embedding,
            name='non_word_embedding', input=config.non_word_char_input_name)
    non_word_conv = build_convolutional_layer(config,
            n_filters=config.n_char_filters,
            filter_width=config.char_filter_width)
    non_word_conv.trainable = config.train_filters
    graph.add_node(non_word_conv, name='non_word_conv', input='non_word_embedding')
    non_word_prev_layer = add_bn_relu(graph, config, 'non_word_conv')
    non_word_pool = build_pooling_layer(config,
            input_width=config.non_word_char_input_width,
            filter_width=config.char_filter_width)
    graph.add_node(non_word_pool,
            name='non_word_pool', input=non_word_prev_layer)
    graph.add_node(Flatten(), name='non_word_flatten', input='non_word_pool')
    prev_non_word_layer = 'non_word_flatten'

    # Word-level input for the context of the non-word error.
    graph.add_input(config.context_input_name,
            input_shape=(config.context_input_width,), dtype='int')
    context_embedding = build_embedding_layer(config,
            input_width=config.context_input_width,
            n_embeddings=config.n_context_embeddings,
            n_embed_dims=config.n_context_embed_dims)
    graph.add_node(context_embedding,
            name='context_embedding', input=config.context_input_name)
    context_conv = build_convolutional_layer(config,
            n_filters=config.n_context_filters,
            filter_width=config.context_filter_width)
    context_conv.trainable = config.train_filters
    graph.add_node(context_conv, name='context_conv', input='context_embedding')
    context_prev_layer = add_bn_relu(graph, config, 'context_conv')
    context_pool = build_pooling_layer(config,
            input_width=config.context_input_width,
            filter_width=config.context_filter_width)
    graph.add_node(context_pool,
            name='context_pool', input=context_prev_layer)
    graph.add_node(Flatten(), name='context_flatten', input='context_pool')
    prev_context_layer = 'context_flatten'

    if config.pool_merge_mode == 'cos':
        dot_axes = ([1], [1])
    else:
        dot_axes = -1

    # Add some number of fully-connected layers without skip connections.
    prev_layer = None
    for i,n_hidden in enumerate(config.fully_connected):
        layer_name = 'dense%02d' %i
        l = build_dense_layer(config, n_hidden=n_hidden)
        if i == 0:
            graph.add_node(l, name=layer_name,
                    inputs=[prev_non_word_layer, prev_context_layer],
                    merge_mode=config.pool_merge_mode,
                    dot_axes=dot_axes)
        else:
            graph.add_node(l, name=layer_name, input=prev_layer)
        prev_layer = layer_name
        if config.batch_normalization:
            graph.add_node(BatchNormalization(), name=layer_name+'bn', input=prev_layer)
            prev_layer = layer_name+'bn'
        graph.add_node(Activation('relu'), name=layer_name+'relu', input=prev_layer)
        if config.dropout_fc_p > 0.:
            graph.add_node(Dropout(config.dropout_fc_p), name=layer_name+'do', input=prev_layer)
            prev_layer = layer_name+'do'

        #prev_layer = layer_name
    
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

    if hasattr(config, 'n_hsm_classes'):
        graph.add_node(build_hierarchical_softmax_layer(config),
            name='softmax', input=prev_layer)
    else:
        graph.add_node(Dense(n_classes, init=config.dense_init,
            W_constraint=maxnorm(config.softmax_max_norm)),
            name='softmax', input=prev_layer)
        prev_layer = 'softmax'
        if config.batch_normalization:
            graph.add_node(BatchNormalization(), name='softmax_bn', input='softmax')
            prev_layer = 'softmax_bn'
        graph.add_node(Activation('softmax'), name='softmax_activation', input=prev_layer)

    graph.add_output(name='multiclass_correction_target', input='softmax_activation')

    load_weights(config, graph)

    optimizer = build_optimizer(config)

    graph.compile(loss={'multiclass_correction_target': config.loss}, optimizer=optimizer)

    return graph


def build_transformers(config, vocabulary):
    non_word_generator = globals()[config.non_word_generator](
            min_edits=config.min_edits,
            max_edits=config.max_edits)

    non_word_input_transformer = MulticlassNonwordTransformer(
            output_width=config.non_word_char_input_width)

    context_input_transformer = MulticlassContextTransformer(
            vocabulary=vocabulary,
            output_width=config.context_input_width,
            unk_index=int(config.context_input_width/2))

    return (non_word_generator,
            non_word_input_transformer,
            context_input_transformer)

def build_vocabulary(word_to_context):
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
    return vocabulary

def load_word_to_context(config):
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

    return word_to_context

def build_generators(word_to_context, vocabulary, config):
    splitter = MulticlassSplitter(
            word_to_context=word_to_context,
            random_state=config.random_state)

    train_data, valid_data, test_data = splitter.split()

    def dataset_size(dataset):
        return sum([len(v) for v in dataset.values()])

    print('train %d validation %d test %d' % (
        dataset_size(train_data),
        dataset_size(valid_data),
        dataset_size(test_data)))

    non_word_generator, non_word_input_transformer, \
            context_input_transformer = build_transformers(
                    config, vocabulary)

    # Fit the label encoder only over the words we're correcting.
    label_encoder = build_label_encoder(
            list(word_to_context.keys()))

    train_generator = MulticlassGenerator(train_data,
            non_word_generator,
            non_word_input_transformer,
            context_input_transformer,
            label_encoder,
            non_word_char_input_name=config.non_word_char_input_name,
            context_input_name=config.context_input_name,
            target_name=config.target_name,
            n_classes=len(word_to_context.keys()))

    valid_generator = MulticlassGenerator(valid_data,
            non_word_generator,
            non_word_input_transformer,
            context_input_transformer,
            label_encoder,
            non_word_char_input_name=config.non_word_char_input_name,
            context_input_name=config.context_input_name,
            target_name=config.target_name,
            n_classes=len(word_to_context.keys()))

    test_generator = MulticlassGenerator(test_data,
            non_word_generator,
            non_word_input_transformer,
            context_input_transformer,
            label_encoder,
            non_word_char_input_name=config.non_word_char_input_name,
            context_input_name=config.context_input_name,
            target_name=config.target_name,
            n_classes=len(word_to_context.keys()))

    return train_generator, valid_generator, test_generator

def build_label_encoder(words):
    label_encoder = LabelEncoder()
    label_encoder.fit(words)
    return label_encoder

"""
Load the Aspell English vocabulary and use it to initialize the retriever
that we use to evaluate our model's performance on the validation set.
We expect the model to surpass the performance of the retriever, as the
model exploits the context of the error.  The retriever functions, then,
as a weak baseline.
"""
def load_aspell_vocabulary(csv_path='~/proj/spelling/data/aspell-dict.csv.gz'):
    df = pd.read_csv(csv_path, sep='\t', encoding='utf8')
    vocabulary = [word for word in df.word.tolist() if "'" not in word and len(word) >= 3 and len(word) <= 7]
    return vocabulary

retriever_lock = threading.Lock()

def build_retriever(vocabulary=None):
    if vocabulary is None:
        vocabulary = load_aspell_vocabulary()

    with retriever_lock:
        aspell_retriever = spelldict.AspellRetriever()
        edit_distance_retriever = spelldict.EditDistanceRetriever(vocabulary)
        retriever = spelldict.RetrieverCollection([aspell_retriever, edit_distance_retriever])
        retriever = spelldict.CachingRetriever(retriever,
            cache_dir='/localwork/ndronen/spelling/spelling_error_cache/')
        jaro_sorter = spelldict.DistanceSorter('jaro_winkler')
        return spelldict.SortingRetriever(retriever, jaro_sorter)

def prepare_data(config):
    word_to_context = load_word_to_context(config)
    vocabulary = build_vocabulary(word_to_context)
    train_generator, valid_generator, test_generator = \
            build_generators(word_to_context, vocabulary, config)

    return word_to_context, vocabulary, \
            train_generator, valid_generator, test_generator

def fit(config):
    word_to_context, vocabulary, \
            train_generator, valid_generator, test_generator = \
                prepare_data(config)

    # We don't know the number of context embeddings in advance, so we
    # set them at runtime based on the size of the vocabulary.
    config.n_context_embeddings = len(vocabulary)
    print('n_context_embeddings %d' % config.n_context_embeddings)

    n_classes = len(word_to_context.keys())
    print('n_classes %d' % n_classes)

    graph = build_model(config, n_classes)

    config.logger('model has %d parameters' % graph.count_params())

    # Violate encapsulation a bit.
    label_encoder = train_generator.label_encoder
    target_map = dict(zip(
        label_encoder.classes_, range(len(label_encoder.classes_))))

    config.logger('building callbacks')

    callbacks = build_callbacks(config,
            valid_generator,
            n_samples=config.n_val_samples,
            dictionary=build_retriever(),
            target_map=target_map)

    # Don't use class weights here; the targets are balanced.
    class_weight = {}

    verbose = 2 if 'background' in config.mode else 1

    graph.fit_generator(train_generator.generate(train=True),
            samples_per_epoch=config.samples_per_epoch,
            nb_worker=config.n_worker,
            nb_epoch=config.n_epoch,
            validation_data=valid_generator.generate(exhaustive=True),
            nb_val_samples=config.n_val_samples,
            callbacks=callbacks,
            class_weight=class_weight,
            verbose=verbose)

###########################################################################
# Callbacks
###########################################################################

class MetricsCallback(keras.callbacks.Callback):
    def __init__(self, config, generator, n_samples, dictionary, target_map):
        self.__dict__.update(locals())
        del self.self

    def on_epoch_end(self, epoch, logs={}):
        correct = []
        y = []
        y_hat = []
        y_hat_dictionary = []

        counter = 0
        pbar = build_progressbar(self.n_samples)
        print('\n')
        g = self.generator.generate(exhaustive=False, train=False)
        n_failed = 0
        while True:
            pbar.update(counter)
            try:
                next_batch = next(g)
            except StopIteration:
                break

            assert isinstance(next_batch, dict)

            # The dictionary's predictions.  Get these first, so we can
            # skip any that the dictionary doesn't have suggestions for.
            # This is to ensure that the evaluation occurs on even ground.
            non_words = next_batch['non_word']
            correct_words = next_batch['correct_word']
            failed = []
            for i,non_word in enumerate(non_words):
                suggestions = self.dictionary[str(non_word)]
                try:
                    suggestion = suggestions[0]
                    target = self.target_map[suggestion]
                    if target is None:
                        raise ValueError('target is None for %s => %s' % (non_word, suggestion))
                    y_hat_dictionary.append(target)
                except IndexError:
                    # I don't know what to do if the dictionary doesn't
                    # offer any suggestions.
                    failed.append(True)
                except KeyError as e:
                    # Or if we don't have a target for the suggested replacement.
                    failed.append(True)

            if any(failed):
                n_failed += len(failed)
                continue

            # The gold standard.
            targets = next_batch[self.config.target_name]
            y.append(np.argmax(targets, axis=1))

            # The model's predictions.
            pred = self.model.predict(next_batch, verbose=0)[self.config.target_name]
            y_hat.append(np.argmax(pred, axis=1))

            counter += len(targets)
            if counter >= self.n_samples:
                print('%d >= %d - stopping loop' % (counter, self.n_samples))
                break

        pbar.finish()

        self.config.logger('\n%d dictionary lookups failed reporting results for %d examples\n' %
                    (n_failed, len(y)))

        self.config.logger('\n')
        self.config.logger('Dictionary')
        self.config.logger('accuracy %.04f F1 %0.4f' %
            (accuracy_score(y, y_hat_dictionary), f1_score(y, y_hat_dictionary, average='weighted')))

        self.config.logger('\n')
        self.config.logger('ConvNet')
        self.config.logger('accuracy %.04f F1 %0.4f\n' %
            (accuracy_score(y, y_hat), f1_score(y, y_hat, average='weighted')))
        self.config.logger('\n')

def build_callbacks(config, generator, n_samples, dictionary, target_map):
    callbacks = []
    mc = MetricsCallback(config, generator, n_samples, dictionary, target_map)
    wn = DenseWeightNormCallback(config)
    es = keras.callbacks.EarlyStopping(patience=config.patience, verbose=1)
    callbacks.extend([mc, wn, es])
    if 'persistent' in config.mode:
        cp = keras.callbacks.ModelCheckpoint(
                filepath=config.model_path + 'model.h5',
                save_best_only=True)
        callbacks.append(cp)

    return callbacks
