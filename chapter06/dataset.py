import unittest

import os
import random
import re
import pickle

import numpy as np

from sklearn.utils import check_random_state
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

import spelling
import spelling.mitton
import spelling.edits
from spelling.jobs import BuildEditDatabaseFromMittonCorpora
from spelling.preprocess import build_char_matrix
from spelling.utils import build_progressbar as build_pbar

from modeling.utils import balanced_class_weights

from keras.utils import np_utils

###########################################################################
# Transformers that create examples of spelling errors.
###########################################################################

class RandomWordTransformer(object):
    def __init__(self, min_edits=1, max_edits=1, random_state=17):
        self.editor = spelling.edits.Editor()
        self.n_edits = [e for e in range(min_edits, max_edits+1)]
        self.random_state = random.Random(random_state)

    def transform(self, X):
        transformed = []
        for x in X:
            n_edits = self.random_state.choice(self.n_edits)
            for i in range(n_edits):
                exes = list(self.editor.edits(x))
                x = self.random_state.choice(exes)
            transformed.append(x)
        return transformed


class LearnedEditTransformer(object):
    def __init__(self, min_edits=1, max_edits=1, n_attempts=1, random_state=17):
        self.__dict__.update(locals())
        self.random_state = check_random_state(random_state)

        self.spelling_path = os.path.dirname(spelling.__path__[0])

        self.corpora = [os.path.join(self.spelling_path, c) for c
                in spelling.mitton.CORPORA]
        self.job = BuildEditDatabaseFromMittonCorpora(self.corpora)
        self.edit_db = self.job.run()
        self.finder = spelling.edits.EditFinder()

    def transform(self, X):
        transformed = []
        for word in X:
            non_word = None
            for i in range(self.n_attempts):
                candidate_non_word = self.transform_word_(word)
                if non_word in self.edit_db.errors:
                    next
                non_word = candidate_non_word
                break
            # Let the client decide what to do with possible None.
            transformed.append(non_word)
        return transformed

    def transform_word_(self, word):
        possible_edits, probs = self.find_possible_edits_(word)
        n_edits = self.sample_number_of_edits_()
        edit_idx = self.sample_edit_idx_(probs, n_edits)

        edit = []
        seen_edits = set()
        for i in edit_idx:
            pe = possible_edits[i]
            if pe in seen_edits:
                continue
            seen_edits.add(pe)
            edit.append(pe)

        if len(edit) == 0:
            raise ValueError('No edits can be made to "%s"' % word)

        return self.finder.apply(word, edit)

    def find_possible_edits_(self, word):
        # Find all the edits we can make to this word.
        possible_edits = list()
        probs = list()

        for subseq in spelling.edits.subsequences(word):
            for e in self.edit_db.edits(subseq):
                _, error_subseq, count = e
                possible_edit = (subseq, error_subseq)
                if count > 0:
                    possible_edits.append(possible_edit)
                    probs.append(count)

        if len(possible_edits) == 0:
            raise ValueError('No edits can be made to "%s"' % word)

        probs = np.array(probs)
        probs = probs / float(probs.sum())

        return possible_edits, probs

    def sample_number_of_edits_(self):
        n_edits_range = np.arange(self.min_edits, self.max_edits+1)
        n_edits_probs = 1. / n_edits_range
        n_edits_probs /= n_edits_probs.sum()
        n_edits = self.random_state.choice(n_edits_range,
                size=1, replace=False, p=n_edits_probs)[0]
        return n_edits

    def sample_edit_idx_(self, probs, n_edits):
        # Sample edits with probability proportional to frequency.
        edit_idx = self.random_state.choice(
                len(probs),
                size=n_edits,
                replace=False,
                p=probs)
        return edit_idx

###########################################################################
# Data set loaders
###########################################################################

class Loader(object):
    """
    Loads a dataset.
    """
    def load(self):
        raise NotImplementedError()

class MulticlassLoader(Loader):
    def __init__(self, pickle_path):
        self.pickle_path = pickle_path

    def load(self):
        with open(self.pickle_path, "rb") as f:
            data = pickle.load(f)
            words = None
            words_to_context = None
            if isinstance(data, tuple):
                words, words_to_context = data
            elif isinstance(data, dict):
                words_to_context = data
            else:
                raise ValueError("Expecting %s to be either tuple or dict, not %s" %
                        (self.pickle_path, type(data)))
            return words, words_to_context
                

###########################################################################
# Data set splitters
###########################################################################

class Splitter(object):
    def split(self):
        raise NotImplementedError()

class MulticlassSplitter(Splitter):
    def __init__(self, word_to_context, train_size=0.9, validation_size=0.5, random_state=17, **kwargs):
        self.__dict__.update(locals())
        self.random_state = check_random_state(random_state)

    def split(self):
        train_data = {}
        validation_data = {}
        test_data = {}

        for word,contexts in self.word_to_context.items():
            train_contexts, other_contexts = train_test_split(
                    contexts, 
                    train_size=self.train_size,
                    random_state=self.random_state)
            validation_contexts, test_contexts = train_test_split(
                    other_contexts,
                    train_size=self.validation_size,
                    random_state=self.random_state)

            train_data[word] = train_contexts
            validation_data[word] = validation_contexts
            test_data[word] = test_contexts

        return train_data, validation_data, test_data

###########################################################################
# Data set tokenizers
###########################################################################

class Tokenizer(object):
    def transform(self, X, y=None):
        pass

class ContextTokenizer(object):
    def transform(self, X, y=None):
        tokenized = []
        for x in X:
            x = x.lower()
            x = re.sub(r'[~`!@#$%^&*()_+={}\[\]<>,.:;?/!\'"|\\-]', " ", x)
            x = x.strip()
            x = re.split(r'\s+', x)
            tokenized.append(x)
        return tokenized

class TestContextTokenizer(unittest.TestCase):
    def test_simple(self):
        toker = ContextTokenizer()
        tokenized = toker.transform(["this is a test - how well does this work?"])
        self.assertEqual(1, len(tokenized))
        self.assertEqual(["this", "is", "a", "test", "how", "well", "does", "this", "work"], tokenized[0])

class ContextWindowTransformer(object):
    def __init__(self, window_size):
        assert window_size % 2 == 1
        self.window_size = window_size
        self.window_center = int(self.window_size/2)

    def transform(self, X, y=None):
        assert y is not None
        windows = []

        assert isinstance(X, (list,tuple))

        for i,x in enumerate(X):
            word = y[i]
            window = [''] * self.window_size

            offset = x.index(word)
            for window_position in range(self.window_size):
                try:
                    x_position = offset + window_position - self.window_center
                    if x_position < 0:
                        raise IndexError()
                    window[window_position] = x[x_position]
                except IndexError:
                    pass

            windows.append(window)

        return windows


class TestContextWindowTransformer(unittest.TestCase):
    def test_simple(self):
        builder = ContextWindowTransformer(5)

        X = [['_', '_', '_', 'this', 'is', 'a', 'test', 'case', '_', '_']]
        y = ["a"]
        actual = builder.transform(X, y)
        expected = [['this', 'is', 'a', 'test', 'case']]
        self.assertEqual(1, len(actual))
        self.assertEqual(expected[0], actual[0])

        X = [['a', 'test', 'case', '_', '_']]
        y = ["a"]
        actual = builder.transform(X, y)
        expected = [['', '', 'a', 'test', 'case']]
        self.assertEqual(1, len(actual))
        self.assertEqual(expected[0], actual[0])


###########################################################################
# Data set transformers
###########################################################################

class Transformer(object):
    """
    Transforms examples into a format consumable by a model.
    """
    def fit(X, y=None, **fit_params):
        raise NotImplementedError()

    def fit_transform(X, y=None, **fit_params):
        raise NotImplementedError()

    def transform(X, y=None):
        raise NotImplementedError()

class MulticlassContextTransformer(Transformer):
    """
    Transforms a list of contexts into a matrix of integers.  A context
    is a list of tokens; the tokens are assumed to be preprocessed as
    needed by the application.
    """
    def __init__(self, vocabulary, output_width, unk_token="<UNK>", unk_index=-1):
        self.__dict__.update(locals())

    def transform(self, X, y=None):
        transformed = np.zeros((len(X), self.output_width))
        for i in range(len(X)):
            x = X[i]
            for j,token in enumerate(x):
                if j == self.unk_index:
                    transformed[i,j] = self.vocabulary[self.unk_token]
                else:
                    try:
                        transformed[i,j] = self.vocabulary[token]
                    except KeyError:
                        # We use the blank word here instead of <UNK>, because
                        # <UNK> denotes the misspelled word.  (Poor choice of names.)
                        transformed[i,j] = self.vocabulary['']
        return transformed

class TestMulticlassContextTransformer(unittest.TestCase):
    def test_simple(self):
        vocabulary = {
                "": 0,
                "this": 1,
                "is": 2,
                "a": 3,
                "test": 4,
                "case": 5,
                "<UNK>": 6
                }
        output_width = 5
        transformer = MulticlassContextTransformer(
                vocabulary, output_width, unk_index=2)
        X = [["this", "is", "a", "test", "case"]]
        actual = transformer.transform(X)
        expected = [[1, 2, 6, 4, 5]]
        self.assertEqual(1, len(actual))
        self.assertEqual(len(expected[0]), len(actual[0]))

class MulticlassNonwordTransformer(Transformer):
    """
    Transforms a list of strings into a matrix of integers.
    """
    def __init__(self, output_width, start_of_word_marker='^', end_of_word_marker='$'):
        self.__dict__.update(locals())

    def transform(self, X, y=None):
        assert all([len(x) <= self.output_width for x in X])
        X = [self.mark(word) for word in X]
        return build_char_matrix(X, width=self.output_width, return_mask=False)

    def mark(self, word):
        return self.start_of_word_marker + word + self.end_of_word_marker

###########################################################################
# Data set generators
###########################################################################

class Generator(object):
    def generate(exhaustive=False, train=False):
        raise NotImplementedError()

class MulticlassGenerator(Generator):
    def __init__(self, word_to_context, non_word_generator, char_input_transformer, context_input_transformer, label_encoder, non_word_char_input_name, context_input_name, target_name, n_classes=None, random_state=17):
        self.__dict__.update(locals())
        self.random_state = random.Random(random_state)

    def generate(self, exhaustive=False, train=False):
        if exhaustive:
            return self.generate_exhaustive(train=train)
        else:
            return self.generate_infinite(train=train)

    def generate_exhaustive(self, train=False):
        words = sorted(list(self.word_to_context.keys()))
        pbar = build_pbar(words)
        for i,word in enumerate(words):
            pbar.update(i+1)
            for context in self.word_to_context[word]:
                yield self.build_next(word, context)
        pbar.finish()

    def generate_infinite(self, train=False):
        words = sorted(list(self.word_to_context.keys()))
        while True:
            for word in words:
                contexts = self.word_to_context[word]
                try:
                    context = self.random_state.choice(contexts)
                except IndexError:
                    continue

                yield self.build_next(word, context)

    def build_next(self, word, context):
        non_word = self.non_word_generator.transform([word])
        non_word_input = self.char_input_transformer.transform(non_word)
        context_input = self.context_input_transformer.transform([context])
        target = self.label_encoder.transform(word)
        if self.n_classes is not None:
            target = np_utils.to_categorical([target], self.n_classes)
        return {
                'correct_word': np.array([word]),
                'non_word': np.array(non_word),
                'context': np.array([context]),
                self.non_word_char_input_name: non_word_input,
                self.context_input_name: context_input,
                self.target_name: target
                }



class BinaryGenerator(Generator):
    def __init__(self, word_to_context, non_word_generator, char_input_transformer, real_word_input_transformer, context_input_transformer, non_word_char_input_name, real_word_char_input_name, real_word_input_name, context_input_name, target_name, retriever, n_classes=None, sample_weight_exponent=1, random_state=17):
        self.__dict__.update(locals())
        self.random_state = random.Random(random_state)

    def generate(self, exhaustive=False, train=False):
        words = sorted(list(self.word_to_context.keys()))
        while True:
            for word in words:
                # Sample a context, and replace the center word with each candidate.
                contexts = self.word_to_context[word]
                try:
                    context = self.random_state.choice(contexts)
                except IndexError:
                    continue
                non_word = self.non_word_generator.transform([word])
                candidates = self.retriever[non_word[0]]

                non_word_char_input = self.char_input_transformer.transform(
                        non_word * len(candidates))
                real_word_char_input = self.char_input_transformer.transform(
                        candidates)

                targets = []
                modified_contexts = []
                for candidate in candidates:
                    targets.append(1 if candidate == word else 0)
                    candidate_context = list(context)
                    candidate_context[int(len(candidate_context)/2)] = candidate
                    modified_contexts.append(candidate_context)
                context_input = self.context_input_transformer.transform(
                        modified_contexts)

                context_input_01 = []
                context_input_02 = []
                context_input_03 = []
                context_input_04 = []
                context_input_05 = []
                for i,ctx_input in enumerate(context_input):
                    context_input_01.append([ctx_input[0]])
                    context_input_02.append([ctx_input[1]])
                    context_input_03.append([ctx_input[2]])
                    context_input_04.append([ctx_input[3]])
                    context_input_05.append([ctx_input[4]])

                real_word_input = self.real_word_input_transformer.transform(
                        # The transformer expects each example to be a list.
                        [[c] for c in candidates])

                targets = np_utils.to_categorical(targets, 2)

                data_dict = {
                        'correct_word': np.array([word] * len(candidates)),
                        'non_word': np.array([non_word[0]] * len(candidates)),
                        'candidate_word': np.array(candidates),
                        self.real_word_char_input_name: real_word_char_input,
                        self.non_word_char_input_name: non_word_char_input,
                        self.real_word_input_name: real_word_input,
                        self.context_input_name: context_input,
                        '%s_%02d' % (self.context_input_name,1): np.array(context_input_01),
                        '%s_%02d' % (self.context_input_name,2): np.array(context_input_02),
                        '%s_%02d' % (self.context_input_name,3): np.array(context_input_03),
                        '%s_%02d' % (self.context_input_name,4): np.array(context_input_04),
                        '%s_%02d' % (self.context_input_name,5): np.array(context_input_05),
                        self.target_name: np.array(targets)
                        }

                class_weights = balanced_class_weights(
                        targets[:, 1].astype(int),
                        n_classes=2,
                        class_weight_exponent=self.sample_weight_exponent)
                sample_weights = np.zeros(len(candidates))
                for k, candidate in enumerate(candidates):
                    sample_weights[k] = class_weights[targets[k, 1]]

                sample_weight_dict = {
                        self.target_name: sample_weights 
                        }

                yield data_dict, sample_weight_dict

if __name__ == '__main__':
    unittest.main()
