import unittest
import traceback

import sys
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
            words_to_context = None
            if isinstance(data, tuple):
                _, words_to_context = data
            elif isinstance(data, dict):
                words_to_context = data
            else:
                raise ValueError("Expecting %s to be either tuple or dict, not %s" %
                        (self.pickle_path, type(data)))
            return words_to_context
                

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
# Data set filters
###########################################################################

class Filter(object):
    def transform(self, X, y=None):
        raise NotImplementedError()

class DigitFilter(Filter):
    def transform(self, X, y=None):
        kept = []
        for context in X:
            if 'digit' in context:
                continue
            kept.append(context)
        return kept

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
    NON_WORD = "<NONWORD>"
    UNKNOWN_WORD = "<UNK>"

    """
    Transforms a list of contexts into a matrix of integers.  A context
    is a list of tokens; the tokens are assumed to be preprocessed as
    needed by the application.
    """
    def __init__(self, vocabulary, output_width, non_word_index=-1):
        self.__dict__.update(locals())

    def transform(self, X, y=None):
        transformed = np.zeros((len(X), self.output_width))
        for i in range(len(X)):
            x = X[i]
            for j,token in enumerate(x):
                if j == self.non_word_index:
                    transformed[i,j] = self.vocabulary[MulticlassContextTransformer.NON_WORD]
                else:
                    try:
                        transformed[i,j] = self.vocabulary[token]
                    except KeyError:
                        # TODO: stop using '<UNK>' to denote the misspelled word.
                        # We use the blank word here instead of <UNK>, because
                        # <UNK> denotes the misspelled word.  (Poor choice of names.)
                        transformed[i,j] = self.vocabulary[MulticlassContextTransformer.UNKNOWN_WORD]
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
                MulticlassContextTransformer.NON_WORD: 6
                }
        output_width = 5
        transformer = MulticlassContextTransformer(
                vocabulary, output_width, non_word_index=2)
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
    def __init__(self, word_to_context, non_word_generator, char_input_transformer, context_input_transformer, label_encoder, non_word_char_input_name, context_input_name, target_name, batch_size=1, n_classes=None, random_state=17):
        self.__dict__.update(locals())
        self.random_state = check_random_state(random_state)

    def generate(self, exhaustive=False, train=False):
        if exhaustive:
            return self.generate_exhaustive(train=train)
        else:
            return self.generate_infinite(train=train)

    def generate_exhaustive(self, train=False):
        words = sorted(list(self.word_to_context.keys()))
        pbar = build_pbar(words)
        while True:
            for i,word in enumerate(words):
                pbar.update(i+1)
                word_list = []
                word_list.append(word)
                for context in self.word_to_context[word]:
                    context_list = []
                    context_list.append(context)
                    try:
                        yield self.build_next(word_list, context_list)
                    except ValueError as e:
                        if str(e) == "no contexts found":
                            continue
                        raise e
                    except Exception as e:
                        print('generate_exhaustive', word_list, context_list, e, type(e))
                        print('')
                        (t, v, tb) = sys.exc_info()
                        traceback.print_tb(tb)
                        print('')
            pbar.finish()

    def generate_infinite(self, train=False):
        words = np.array(list(self.word_to_context.keys()))
        while True:
            word_sample = self.random_state.choice(words,
                    size=self.batch_size)
            try:
                yield self.build_next(word_sample)
            except ValueError as e:
                if str(e) == "no contexts found":
                    continue
                raise e
            except Exception as e:
                print('generate_infinite', word_sample, e, type(e))
                print('')
                (t, v, tb) = sys.exc_info()
                traceback.print_tb(tb)
                print('')

    def build_next(self, words, ctx=None):
        non_words = []
        non_word_inputs = []
        contexts = []
        context_inputs = []
        context_inputs_01 = []
        context_inputs_02 = []
        context_inputs_03 = []
        context_inputs_04 = []
        context_inputs_05 = []
        targets = []

        kept_words = []

        for i,word in enumerate(words):
            if ctx is None:
                ctxs = self.word_to_context[word]
                if len(ctxs) == 0:
                    continue
                j = self.random_state.choice(len(ctxs))
                context = ctxs[j]
            else:
                context = ctx[i]

            kept_words.append(word)

            non_word = self.non_word_generator.transform([word])
            non_word_input = self.char_input_transformer.transform(non_word)
            context_input = self.context_input_transformer.transform([context])

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

            target = self.label_encoder.transform(word)
            if self.n_classes is not None:
                target = np_utils.to_categorical([target], self.n_classes)

            non_words.append(non_word)
            non_word_inputs.append(non_word_input)
            contexts.append(np.array(context))
            context_inputs.append(context_input)
            context_inputs_01.append(context_input_01)
            context_inputs_02.append(context_input_02)
            context_inputs_03.append(context_input_03)
            context_inputs_04.append(context_input_04)
            context_inputs_05.append(context_input_05)
            targets.append(target)

        if len(kept_words) == 0:
            raise ValueError("no contexts found")

        non_words = np.array(non_words)
        non_word_inputs = np.concatenate(non_word_inputs)

        context_inputs = np.concatenate(context_inputs)
        context_inputs_01 = np.concatenate(context_inputs_01)
        context_inputs_02 = np.concatenate(context_inputs_02)
        context_inputs_03 = np.concatenate(context_inputs_03)
        context_inputs_04 = np.concatenate(context_inputs_04)
        context_inputs_05 = np.concatenate(context_inputs_05)
        targets = np.concatenate(targets)

        return {
                'correct_word': np.array(kept_words),
                'non_word': non_words,
                self.non_word_char_input_name: non_word_inputs,
                self.context_input_name: context_inputs,
                '%s_%02d' % (self.context_input_name,1): context_inputs_01,
                '%s_%02d' % (self.context_input_name,2): context_inputs_02,
                '%s_%02d' % (self.context_input_name,3): context_inputs_03,
                '%s_%02d' % (self.context_input_name,4): context_inputs_04,
                '%s_%02d' % (self.context_input_name,5): context_inputs_05,
                self.target_name: targets
                }

class BinaryGenerator(Generator):
    def __init__(self, word_to_context, non_word_generator, char_input_transformer, real_word_input_transformer, context_input_transformer, non_word_char_input_name, real_word_char_input_name, real_word_input_name, context_input_name, target_name, retriever, n_classes=None, sample_weight_exponent=1, random_state=17, batch_size=1, use_real_word_examples=False):
        self.__dict__.update(locals())
        self.random_state = check_random_state(random_state)

    def generate(self, exhaustive=False, train=False):
        if exhaustive:
            return self.generate_exhaustive(train=train)
        else:
            return self.generate_infinite(train=train)

    def generate_infinite(self, train=False):
        words = list(sorted(self.word_to_context.keys()))
        self.random_state.shuffle(words)
        while True:
            word_sample = self.random_state.choice(words,
                    size=self.batch_size)
            try:
                yield self.build_next(word_sample)
            except ValueError as e:
                if str(e) == "no contexts found":
                    continue
                raise e
            except Exception as e:
                print('generate_infinite', word_sample, e, type(e))
                print('')
                (t, v, tb) = sys.exc_info()
                traceback.print_tb(tb)
                print('')

    def generate_exhaustive(self, train=False):
        words = sorted(list(self.word_to_context.keys()))
        pbar = build_pbar(words)
        print('generate_exhaustive')
        while True:
            for i,word in enumerate(words):
                pbar.update(i+1)
                word_list = []
                word_list.append(word)
                for context in self.word_to_context[word]:
                    context_list = []
                    context_list.append(context)
                    try:
                        yield self.build_next(word_list, context_list)
                    except ValueError as e:
                        if str(e) == "no contexts found":
                            continue
                        raise e
                    except Exception as e:
                        print('generate_exhaustive', word_list, context_list, e, type(e))
                        print('')
                        (t, v, tb) = sys.exc_info()
                        traceback.print_tb(tb)
                        print('')
            pbar.finish()

    def build_next(self, words, ctx=None):
        correct_words = []
        non_words = []
        non_word_char_inputs = []
        real_words = []
        real_word_char_inputs = []
        modified_contexts = []
        contexts = []
        context_inputs = []
        context_inputs_01 = []
        context_inputs_02 = []
        context_inputs_03 = []
        context_inputs_04 = []
        context_inputs_05 = []
        targets = []
        sample_weights = []

        kept_words = []

        if isinstance(words, str):
            # Batch size is 1 and a single word was passed in.
            words = [words]

        for i,word in enumerate(words):
            # Sample a context, and replace the center word with each candidate.
            try:
                if ctx is None:
                    ctxs = self.word_to_context[word]
                    if len(ctxs) == 0:
                        continue
                    j = self.random_state.choice(len(ctxs))
                    context = ctxs[j]
                else:
                    context = ctx[i]
            except IndexError:
                print('index error - skipping word %s' % word)
                continue

            kept_words.append(word)

            # TODO: the generator sometimes creates real words.
            non_word = self.non_word_generator.transform([word])[0]
            # TODO: does it matter whether the candidate list contains
            # the non-word?
            candidates = self.retriever[non_word]
            # This ensures that there's always an example in a mini-batch
            # with target 1.  
            if word not in candidates:
                candidates.append(word)

            # Add the candidates to the batch.
            correct_words.extend([word] * len(candidates))
            non_words.extend([non_word] * len(candidates))
            real_words.extend(candidates)

            if self.use_real_word_examples:
                # Add to the batch one more example, consisting of the
                # real word itself as an example non-word, and the real
                # word as the candidate and true correction.
                correct_words.append(word)
                non_words.append(word)
                real_words.append(word)
                candidates.append(word)

            candidate_targets = []
            for candidate in candidates:
                candidate_targets.append(1 if candidate == word else 0)
                candidate_context = list(context)
                candidate_context[int(len(candidate_context)/2)] = candidate
                modified_contexts.append(candidate_context)

            targets.extend(candidate_targets)

            candidate_targets = np_utils.to_categorical(candidate_targets, 2)
            class_weights = balanced_class_weights(
                candidate_targets[:, 1].astype(int),
                n_classes=2,
                class_weight_exponent=self.sample_weight_exponent)
            for k, candidate in enumerate(candidates):
                sample_weights.append(
                        class_weights[candidate_targets[k, 1]])

            contexts.append(modified_contexts)

        if len(kept_words) == 0:
            raise ValueError("no contexts found")

        context_inputs = self.context_input_transformer.transform(
            modified_contexts)
       
        for i,ctx_input in enumerate(context_inputs):
            context_inputs_01.append([ctx_input[0]])
            context_inputs_02.append([ctx_input[1]])
            context_inputs_03.append([ctx_input[2]])
            context_inputs_04.append([ctx_input[3]])
            context_inputs_05.append([ctx_input[4]])

        #print("non_words", non_words)
        #print("non_words[0]", non_words[0])
        #print("non_words[-1]", non_words[-1])
        #print("real_words", real_words)
        #print("correct_words", correct_words)

        non_word_char_inputs = self.char_input_transformer.transform(non_words)
        real_word_char_inputs = self.char_input_transformer.transform(real_words)

        # This transformer expects each example to be a list.  (Just this transformer?)
        real_word_inputs = self.real_word_input_transformer.transform(
            [[real_word] for real_word in real_words])

        targets = np_utils.to_categorical(targets, 2)

        data_dict = {
                'correct_word': np.array(correct_words),
                'non_word': np.array(non_words),
                'candidate_word': np.array(real_words),
                self.real_word_char_input_name: np.array(real_word_char_inputs),
                self.non_word_char_input_name: np.array(non_word_char_inputs),
                self.real_word_input_name: np.array(real_word_inputs),
                self.context_input_name: context_inputs,
                '%s_%02d' % (self.context_input_name,1): np.array(context_inputs_01),
                '%s_%02d' % (self.context_input_name,2): np.array(context_inputs_02),
                '%s_%02d' % (self.context_input_name,3): np.array(context_inputs_03),
                '%s_%02d' % (self.context_input_name,4): np.array(context_inputs_04),
                '%s_%02d' % (self.context_input_name,5): np.array(context_inputs_05),
                self.target_name: targets
                }

        sample_weight_dict = {
                self.target_name: np.array(sample_weights)
                }

        return data_dict, sample_weight_dict


if __name__ == '__main__':
    unittest.main()
