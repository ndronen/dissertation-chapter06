# coding: utf-8

import os
import random
import pickle
import collections
import operator
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer

from spelling.utils import build_progressbar as build_pbar
import spelling.dictionary

CONTEXT_CORPUS_PATH = '/work/ndronen/datasets/enwiki-20140903/sents-initial.txt'
DICTIONARY_PATH = '/export/home/ndronen/proj/spelling/data/aspell-dict.csv.gz'

class SpellingErrorContextPreprocessor(object):
    def __init__(self, context_corpus=CONTEXT_CORPUS_PATH, vectorizer=CountVectorizer(), dictionary_path=DICTIONARY_PATH, index_cache=None, context_index=None):
        self.__dict__.update(locals())
        del self.self

        print('loading dictionary')
        self.load_dictionary()
        print('reading contexts')
        self.contexts = self.read_contexts(self.context_corpus)
        print('cleaning contexts')
        self.clean_contexts(self.contexts)
        print('indexing contexts')
        self.index_contexts(self.contexts)
        print('building word-to-context index')
        self.word_to_context_index = self.build_word_to_context_index()

        self.indexed_shorter_words = []
        for short_word in self.shorter_words:
            if short_word in self.word_to_index:
                self.indexed_shorter_words.append(short_word)

        self.indexed_shorter_word_counts = []
        for word in self.indexed_shorter_words:
            self.indexed_shorter_word_counts.append(
                (word, len(self.word_to_context_index[word])))
                            
        # Sort the count of short words contexts.
        self.indexed_shorter_word_counts = sorted(
            self.indexed_shorter_word_counts,
            key=operator.itemgetter(1), reverse=True)
                                            
        # These are the words with >= 1000 contexts.
        # TODO: do this programmatically.
        self.indexed_shorter_word_counts = self.indexed_shorter_word_counts[0:10700]

        # Only keep those words with >= 1000 contexts.
        self.indexed_shorter_words = [word for word,count in
                self.indexed_shorter_word_counts]

    def load_dictionary(self):
        # Since short words are harder to correct with high accuracy, we focus
        # on words of length 3-7.
        self.df = pd.read_csv(self.dictionary_path, sep='\t', encoding='utf8')
        self.df['len'] = self.df.word.apply(len)
        self.dictionary_vocabulary = set(self.df.word.tolist())
        self.shorter_words = self.df[(self.df.len >= 3) & (self.df.len <= 7)].word.tolist()

    def read_contexts(self, context_corpus):
        with open(context_corpus) as f:
            return f.read().split('\n')

    def clean_contexts(self, contexts):
        for i,sentence in enumerate(contexts):
            sentence = sentence.replace('</s>', '')
            sentence = sentence.replace(" 's", "'s")
            sentence = sentence.replace(" 'nt", "'nt")
            contexts[i] = sentence

    def index_contexts(self, contexts):
        if self.context_index is None:
            if self.index_cache is not None and os.path.exists(self.index_cache) and os.stat(self.index_cache).st_size > 0:
                print('loading indices from cache %s' % self.index_cache)
                with open(self.index_cache, "rb") as f:
                    self.vectorizer, self.context_index = pickle.load(f)
            else:
                print('fitting vectorizer on %d sentences' % len(contexts))
                self.context_index = self.vectorizer.fit_transform(contexts)

                if self.index_cache is not None:
                    with open(self.index_cache, "wb") as f:
                        print('dumping indices to cache %s' % self.index_cache)
                        pickle.dump((self.vectorizer, self.context_index), f)
        else:
            assert self.context_index.shape[1] == len(self.vectorizer.vocabulary_)

        self.word_to_index = self.vectorizer.vocabulary_
        self.index_to_word = dict(((v,k) for k,v in self.word_to_index.items()))

    def build_word_to_context_index(self):
        """
        Build a mapping from words to the indices of the contexts in which they appear.
        """
        column_context_index = self.context_index.tocsc()
    
        word_to_context_index = {}
    
        pbar = build_pbar(self.shorter_words)
        for i,word in enumerate(self.shorter_words):
            pbar.update(i+1)
            if word in self.word_to_index:
                col_idx = self.word_to_index[word]
                nonzeros = column_context_index[:, col_idx].nonzero()
                contexts_containing_word = [x for x in nonzeros[0]]
                word_to_context_index[word] = contexts_containing_word
        pbar.finish()
    
        return word_to_context_index
    
    def sample_words(self, word_length_to_word_index, n_words_per_length=100, rng=random.Random(17), retriever=spelling.dictionary.AspellRetriever()):
        """
        Sample `n_words_per_length` words from each key of `word_length_to_word_index`,
        which is a dictionary with word length as keys and a list
        of words of that length as values.   This function ensures
        that the distribution of word lengths is uniform.
        """
        sampled_words = []
    
        for length in word_length_to_word_index.keys():
            words_of_length = list(word_length_to_word_index[length])
            rng.shuffle(words_of_length)
            samples = []
            while len(samples) < n_words_per_length:
                next_word = words_of_length.pop()
                candidate_list = retriever[next_word]
                if len(candidate_list) > 10:
                    samples.append(next_word)
    
            sampled_words.extend(samples)
    
        return sampled_words
    
    def sample_contexts(self, sampled_words, n_contexts_per_word=1000, rng=random.Random(17)):
        """
        Sample `n_contexts_per_word` contexts for each word.
        """
        sampled_word_contexts = collections.defaultdict(list)
    
        for word in sampled_words:
            context_indices = self.word_to_context_index[word]
            rng.shuffle(context_indices)
            for i,idx in enumerate(context_indices):
                context = self.contexts[idx]
                # Require the word to appear alone in the context.
                if ' '+word+' ' in context:
                    sampled_word_contexts[word].append(context)
                if len(sampled_word_contexts[word]) == n_contexts_per_word:
                    break
    
        return sampled_word_contexts
    
    def sample_words_and_contexts(self, n_words_per_length=100, n_contexts_per_word=1000):
        # Index the words by length.
        print('indexing words by length')
        word_length_to_word_index = collections.defaultdict(set)
        for word in self.indexed_shorter_words:
            word_length_to_word_index[len(word)].add(word)
            
        print('sampling %d words of length 3-7 characters' % n_words_per_length)
        # Sample 100 words of each length 3-7 characters.
        sampled_words = self.sample_words(word_length_to_word_index)
        
        # Sample 
        print('sampling %d contexts per word' % n_contexts_per_word)
        sampled_contexts = self.sample_contexts(sampled_words, n_contexts_per_word)
    
        return sampled_words, sampled_contexts

"""
def find_relevant_contexts(preprocessor):
    # Find the words in the dictionary that are in the context index.
    for word in preprocessor.df.word.tolist():
        if word in preprocessor.vectorizer.vocabulary_:
            preprocessor.indexed_vocab.append(word)

    # How many times does each word in the context index occur?
    if len(preprocessor.all_vocab_counts) == 0:
        # Get the non-zero indices of the words in the context index.
        print('getting indices of non-zero indices in context index')
        nz = preprocessor.context_index.nonzero()
        nz_row = nz[0]
        nz_col = nz[1]

        print('counting frequencies of all words in the context index')
        pbar = build_pbar(nz_col)
        for i,col_i in enumerate(nz_col):
            pbar.update(i+1)
            preprocessor.all_vocab_counts[preprocessor.index_to_word[col_i]] += 1
        pbar.finish()

    # In how many contexts do the words that are in both the dictionary
    # and the context index occur?
    if len(preprocessor.indexed_vocab_counts) == 0:
        for word in preprocessor.indexed_vocab:
            preprocessor.indexed_vocab_counts[word] = preprocessor.all_vocab_counts[word]

    # Which words that satisfy our length criteria (3-7 characters)
    # are in the context index?
    if len(preprocessor.indexed_shorter_words) == 0:
        for short_word in preprocessor.shorter_words:
            if short_word in preprocessor.indexed_vocab_counts:
                preprocessor.indexed_shorter_words.append(short_word)
    
    # In how many contexts do those short words occur?
    if len(preprocessor.found_shorter_word_counts) == 0:
        for word in preprocessor.indexed_shorter_words:
            preprocessor.found_shorter_word_counts.append(
                    (word, preprocessor.indexed_vocab_counts[word]))
    
        # Sort the count of short words contexts.
        preprocessor.found_shorter_word_counts = sorted(
                preprocessor.found_shorter_word_counts,
                key=operator.itemgetter(1), reverse=True)
    
        # These are the words with >= 1000 contexts.
        preprocessor.found_shorter_word_counts = preprocessor.found_shorter_word_counts[0:10700]
    
    # Get a histogram of the lenghts of these words.
    if len(preprocessor.lengths) == 0:
        for tup in preprocessor.found_shorter_word_counts:
            preprocessor.lengths[len(tup[0])] += 1
    
    # Count the number of contexts 
    if len(preprocessor.length_counts) == 0:
        for tup in preprocessor.found_shorter_word_counts:
            preprocessor.length_counts[len(tup[0])] += tup[1]
        
    # Report the average number of contexts for words of each length.
    for length in preprocessor.lengths.keys():
        print('length %d average number of contexts %.02f' % (
            length,
            preprocessor.length_counts[length]/float(preprocessor.lengths[length])))
    
    # Require that all of these shorter words occur in the dictionary.
    assert all(f in preprocessor.dictionary_vocabulary for f in preprocessor.indexed_shorter_words)
"""
