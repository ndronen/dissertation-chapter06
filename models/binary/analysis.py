# coding: utf-8

import sys
import collections
import json
import pandas as pd
import numpy as np
import string
import uuid

import modeling.utils
import spelling.evaluate
import spelling.mitton
import spelling.preprocess
import spelling.service as service
from spelling.utils import build_progressbar as build_pbar
from spelling.mitton import evaluate_ranks

import importlib

# TODO: 
# To compare to Google N-Gram spelling correction, I need to take the
# examples for which I was able to obtain corrections from the grammar5
# spelling service and rank them using a ConvNet.  

#BEST_MODEL_DIR = 'models/binary/convolutional_context_deep_residual_plus_isolated/gaussian_sd_0_00'
BEST_MODEL_DIR = 'models/binary/convolutional_context_deep_residual_plus_isolated/gaussian_sd_0_05_random_embeddings'
RANKS = [1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100,200,300,400,500]

def run_init(model_dir):
    config, setup = init(model_dir, config_file='config.json')
    df = generate_test_data(config, setup)
    return config, setup, df

def init(model_dir, config_file='config.json', model_weights=''):
    json_path = model_dir + '/' + config_file
    with open(json_path) as f:
        model_json = json.load(f)
    
    model_json['logger'] = modeling.utils.callable_print
    model_json['validation_size'] = 4500
    if model_weights is not None and len(model_weights):
        model_json['model_weights'] = model_weights

    config = modeling.utils.ModelConfig(**model_json)

    aspell = pd.read_csv('~/proj/spelling/data/aspell-dict.csv.gz', sep='\t', encoding='utf8')

    M = load_model_module(model_dir)
    setup = M.setup(config)
    setup['aspell_vocab'] = aspell.word.tolist()

    return config, setup

def fixup_context(context, aspell_vocab):
    fixed_context = []
    for i,word in enumerate(context):
        try:
            next_word = context[i+1]
        except IndexError:
            next_word = ''

        if word not in aspell_vocab or 'digit' in word:
            # Try to fix the word.
            if word.upper() in aspell_vocab:
                word = word.upper()
            elif word.title() in aspell_vocab:
                word = word.title()
            elif word == 'digit':
                # Are the odds greater that this is in the range 2-9?
                if next_word.endswith('s'):
                    word = '2'
                else:
                    word = '1'
            elif word == 'digitdigit':
                # Just guess a 2-digit number.
                word = '10'
            elif word == 'digitdigitdigit':
                # Just guess a 3-digit number.
                word = '123'
            elif word == 'digitdigitdigitdigit':
                # 4 digits is probably a year.
                word = '1919'
            elif 'digit' in word:
                word = word.replace('digit', '1')
            else:
                # The word can't be fixed.
                raise ValueError()

            # After all of the transformations, require that the word
            # is either in the Aspell vocabulary or is an integer.
            if word not in aspell_vocab:
                # Allow it if it's a number.
                int(word)

        fixed_context.append(word)
    return fixed_context

def generate_test_data(config, setup):
    test_data = setup['test_data']
    retriever = setup['valid_retreiver']
    non_word_generator = setup['non_word_generator']
    aspell_vocab = set(setup['aspell_vocab'])
    
    pbar = build_pbar(test_data)
    
    examples = collections.defaultdict(list)
    
    for i,(word,context) in enumerate(test_data):
        pbar.update(i+1)
    
        try:
            # Require that all of a context's words are in the Aspell
            # English dictionary.
            context = fixup_context(context, aspell_vocab)
        except ValueError:
            continue

        non_words = set()
        for j in range(20):
            # Generate up to 5 non-words for this word.
            if len(non_words) == 5:
                break
    
            non_word = non_word_generator.transform([word])[0]

            if any([c in string.ascii_uppercase for c in non_word]):
                raise ValueError("generated created %s from %s" % (non_word, word))

            if non_word in aspell_vocab:
                continue
            non_words.add(non_word)
    
        for non_word in list(sorted(non_words)):
            new_context = list(context)
            new_context[2] = non_word
            new_context_str = ' '.join(new_context)
    
            examples['word'].append(word)
            examples['non_word'].append(non_word)
            examples['context'].append(new_context_str)
    
    pbar.finish()

    return pd.DataFrame(data=examples)

def get_grammar5_corrections(test_data_df, host):
    url = host + "/pkt-aggregator/Service/spelling"

    words = test_data_df.word.tolist()
    non_words = test_data_df.non_word.tolist()
    contexts = test_data_df.context.tolist()

    dfs = []
    pbar = build_pbar(test_data_df)

    seen = set()

    for i,context in enumerate(contexts):
        pbar.update(i+1)

        word = words[i]
        non_word = non_words[i]

        if non_word in seen:
            continue
        seen.add(non_word)

        try:
            correction_df = pd.DataFrame(service.correct(url, context))
            if len(correction_df) == 0:
                continue
            correction_df = correction_df[correction_df.NonWord == non_word]
            correction_df['word'] = word
            correction_df['non_word'] = non_word
            correction_df['context'] = context
            correction_df['uuid'] = str(uuid.uuid4())
            correction_df['rank'] = correction_df.index.tolist()
            dfs.append(correction_df)
        except TypeError as e:
            print("skipping %s: %s" % (context, e))

    return dfs

def load_config(model_dir):
    return json.load(open(model_dir + '/config.json'))

def load_model_module(model_dir=BEST_MODEL_DIR):
    sys.path.append('.')
    model_module_path = model_dir.replace('/', '.') + '.model'
    return importlib.import_module(model_module_path)

def load_model(model_dir=BEST_MODEL_DIR, model_weights='model.h5'):
    # Load the model and run the data through it.
    model_weights = model_dir + '/' + model_weights
    config, objects = init(model_dir, model_weights=model_weights)

    M = load_model_module(model_dir)
    model = M.build_model(config, n_classes=2)
    return model, config, objects

def rank_candidates(model, example, target_name):
    # Return the order of the maximum probabilities, from greatest to least.
    prob = model.predict(example)[target_name]
    print("prob", prob)
    # Reverse the argsort.
    temp = prob[:, 1].argsort()[::-1]
    rank = np.empty(len(prob), int)
    rank[temp] = np.arange(len(prob))
    assert min(rank) == 0
    assert max(rank) < len(prob)
    return rank

def compute_ranks(dfs, ranks=[1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100]):
    results = collections.defaultdict(list)
    for name in dfs.keys():
        df = dfs[name]
        n = len(df[df['rank'] == 0])
        for rank in ranks:
            m = len(df[(df['rank'] < rank) & (df.candidate == df.word)])
            results['Model'].append(name)
            results['Rank'].append(rank)
            results['N'].append(n)
            results['M'].append(m)
            results['Accuracy'].append(m/float(n))
    return pd.DataFrame(results)

def grammar5_spelling(df, csv_file, model_vocab):
    grammar5_dfs = get_grammar5_corrections(df, "http://grammar5qa.linguisticanxiety.com/")
    grammar5_df = pd.concat(grammar5_dfs)
    grammar5_df['CandidateInModelVocab'] = grammar5_df.Candidate.apply(lambda word: word in model_vocab)
    grammar5_df['candidate'] = grammar5_df.Candidate
    add_columns_for_evaluate_ranks(grammar5_df)

    grammar5_df.to_csv(csv_file, sep='\t', encoding='utf8')
    return grammar5_df

def combine_grammar5_dfs():
    grammar5_dfs = []
    files = get_ipython().magic('%sx ls /tmp/grammar*.csv')
    for f in files:
        grammar5_dfs.append(pd.read_csv(f, sep='\t', encoding='utf8'))

    grammar5_df = pd.concat(grammar5_dfs, ignore_index=True)
    grammar5_df['rank'] = grammar5_df['rank'].astype(int)

    grammar5_df_model_vocab = grammar5_df[grammar5_df.CandidateInModelVocab].copy()

    grammar5_df_model_vocab_grouped = grammar5_df_model_vocab.groupby('uuid')

    groups = grammar5_df_model_vocab_grouped.groups
    pbar = build_pbar(groups)
    for i,(uuid,idx) in enumerate(groups.items()):
        pbar.update(i+1)
        #df_tmp = df_g5_grouped.get_group(name)
        grammar5_df_model_vocab.loc[idx, 'rank'] = range(len(idx))
    pbar.finish()

    #old_ranks = grammar5_df_model_vocab['rank'].values
    #new_ranks = np.zeros(len(grammar5_df_model_vocab)).astype(int)
    #next_val = -1
    #for i,val in enumerate(old_ranks):
    #    if val == 0:
    #        next_val = -1
    #    next_val += 1
    #    new_ranks[i] = next_val

    #grammar5_df_model_vocab['rank'] = new_ranks

    grammar5_df_model_vocab['candidate'] = grammar5_df_model_vocab.Candidate

    add_columns_for_evaluate_ranks(grammar5_df_model_vocab)

    return grammar5_df_model_vocab

def build_model_data(df_g5):
    words = df_g5[df_g5['rank'] == 0].word.tolist()
    non_words = df_g5[df_g5['rank'] == 0].non_word.tolist()
    contexts = df_g5[df_g5['rank'] == 0].context.tolist()
    split_contexts = []

    for i,c in enumerate(contexts):
        word = words[i]
        c = c.split(' ')
        c[2] = word
        split_contexts.append(c)

    data = [[words[i],non_words[i],split_contexts[i]] for i in range(len(words))]
    return data

def rank_with_model(model, generator):
    model_ranks = []
    model_words = []
    model_candidates = []
    model_non_words = []
    model_contexts = []
    model_uuids = []

    while True:
        try:
            next_batch = next(generator)
        except StopIteration:
            break

        model_candidates.extend(next_batch['candidate_word'])
        model_non_words.extend(next_batch['non_word'])
        model_contexts.extend([' '.join(c) for c in next_batch['contexts']])
        model_words.extend(next_batch['correct_word'])

        ranks = rank_candidates(model, next_batch, config.target_name)
        model_ranks.extend(ranks.tolist())

        model_uuids.extend([str(uuid.uuid1())] * len(ranks))

        assert len(model_candidates) == \
                len(model_non_words) == \
                len(model_contexts) == \
                len(model_words) == \
                len(model_ranks)

    df = pd.DataFrame(data={
        'word': model_words,
        'rank': model_ranks,
        'non_word': model_non_words,
        'candidate': model_candidates,
        'context': model_contexts,
        'uuid': model_uuids
        })

    add_columns_for_evaluate_ranks(df)

    return df

def add_columns_for_evaluate_ranks(df):
    # These are expected by spelling.mitton.evaluate_ranks.
    df['correct_word_in_dict'] = True
    df['correct_word'] = df.word
    df['suggestion_index'] = df['rank']
    df['suggestion'] = df.candidate

def rank_correlations(dfs, word_probs, ranks=[1,2,3,4,5]):
    """
    dfs : dict (str: DataFrame)
        A mapping from model names to data frames of ranks.
    word_probs : dict
        A mapping from words to Google N-Gram unigram probabilities.
    """

    columns = ["Model",
            "Type",
            "Spearman's $\rho$", 
            "Spearman's $\rho$ p-value",
            "Kendall's $\tau$",
            "Kendall's $\tau$ p-value"]

    def compute_rank_correlations(accum, x, y):
        x = sp.stats.rankdata(x)
        y = sp.stats.rankdata(y)

        sr = sp.stats.spearmanr(x, y)
        accum[columns[2]].append(sr.correlation)
        accum[columns[3]].append(sr.pvalue)

        kt = sp.stats.kendalltau(x, y)
        accum[columns[4]].append(kt.correlation)
        accum[columns[5]].append(kt.pvalue)

    results = collections.defaultdict(list)

    for model_name,df in dfs.items():
        probs = df.candidate.apply(lambda c: word_probs[c])
        non_zero_probs = probs > 0.
                
        # The rank correlation of the position in the candidate list
        # where a word appears and the word's unigram probability.
        results[columns[0]].append(model_name)
        results[columns[1]].append("All candidate lists")
        compute_rank_correlations(results,
                df['rank'][non_zero_probs],
                probs[non_zero_probs])

        # The rank correlation of the position in the candidate list
        # where a word appears and the word's unigram probability, only
        # for those candidate lists where the candidate at position 0
        # is not the correct word.  This probably requires a groupby.
        """
        uuids = set()
        df_group = df.groupby('uuid')
        pbar = build_pbar(df_group.groups)
        print(model_name)
        for i,(name,idx) in enumerate(df_group.groups.items()):
            pbar.update(i+1)
            df_tmp = df.iloc[idx, :]
            df_top = df_tmp[df_tmp['rank'] == 0]
            if df_top.candidate.tolist()[0] != df_top.correct_word.tolist()[0]:
                uuids.add(name)
        pbar.finish()
        wrong_candidate_lists = df.uuid.isin(uuids)

        results[columns[0]].append(model_name)
        results[columns[1]].append("All candidate lists")
        compute_rank_correlations(results,
                df[wrong_candidate_lists & non_zero_probs]['rank'],
                probs[wrong_candidate_lists & non_zero_probs])
        """

    return pd.DataFrame(data=results)[columns]
