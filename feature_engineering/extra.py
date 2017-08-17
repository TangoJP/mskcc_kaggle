import numpy as np
import pandas as pd
import re
from collections import Counter
from nltk import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

def count_appearances1(list_of_words):
    '''
    Count the frequency of each of the unique words in a list of words.

    INPUT:
    =====
    list_of_words : list
        A list of words, some of which have multiple entries

    OUTPUT:
    =======
    appearances : dict
        A dictionary containing number of appearances in the list for each
        unique word in the list
    '''
    c = Counter(list_of_words)
    appearances = dict(c)

    return appearances

def create_mutation_words_table(tokenized_text, normed=False):
    '''
    Create table for words to describe the mutation types from a list of
    tokenized words

    INPUT:
    ======
    text : list
        a list of tokenized words

    OUTPUT:
    =======
    mutation table : a list of sets
    '''
    # List of words for mutation types
    mutation_patterns = ['truncation', 'deletion', 'promoter','amplification', 'epigenetic', 'frame', 'overexpression',
                     'duplication', 'insertion','subtype', 'fusion', 'splice', 'wildtype']

    appearances = []
    for pattern in mutation_patterns:
        appearance = len([word for word in tokenized_text if pattern in word.lower()])
        appearances.append(appearance)

    if normed == 'mutation_types':
        appearances = np.array(appearances)
        if np.sum(appearances) != 0:
            appearances = appearances / np.sum(appearances)
        table = dict(zip(mutation_patterns, appearances))
    elif normed == 'total_text':
        appearances = np.array(appearances)
        appearances = appearances / len(tokenized_text)
        table = dict(zip(mutation_patterns, appearances))
    else:
        table = dict(zip(mutation_patterns, appearances))
        table['Total'] = np.sum(appearances)

    return table

def convert_mutation_type(data):
    '''
    Convert the 'Variant' Data into mutation_type in a new column, returns the new data with a new column

    Input
    =====
    data : DataFrame
        The train or test data containing Variant information

    Output
    ======
    data : DataFrame
        'mutation_type' is added to the original data from the input
    '''
    # Copy the Variation into a new column (this could be just an empty copy with Nones)
    data['mutation_type'] = data['Variation']

    # Define regex pattern for point mutants
    point_mutation_pattern = \
        r"[ARNDCEQGHILKMFPSTWYV]{1}[0-9]{1,4}[ARNDCEQGHILKMFPSTWYV*]?$"

    # Define new mutation types
    major_types = ['Truncation', 'Point Mutation', 'Deletion', 'Promoter Mutations',
       'Amplification', 'Epigenetic', 'Frame Shift', 'Overexpression',
       'Deletion-Insertion', 'Duplication', 'Insertion',
       'Gene Subtype', 'Fusion', 'Splice', 'Copy Number Loss', 'Wildtype']

    # Convert the Variant information to mutation types
    data.loc[(data['Variation'].str.match(point_mutation_pattern)), 'mutation_type']= 'Point Mutation'
    data.loc[(data['Variation'].str.contains('missense', case=False)), 'mutation_type']= 'Point Mutation'
    data.loc[(data['Variation'].str.contains('fusion', case=False)), 'mutation_type']= 'Fusion'
    data.loc[(data['Variation'].str.contains('deletion', case=False)), 'mutation_type']= 'Deletion'
    data.loc[((data['Variation'].str.contains('del', case=False))\
            &(data['Variation'].str.contains('delins', case=False) == False)),
            'mutation_type']= 'Deletion'
    data.loc[((data['Variation'].str.contains('ins', case=False))\
            &(data['Variation'].str.contains('delins', case=False) == False)),
            'mutation_type']= 'Insertion'
    data.loc[((data['Variation'].str.contains('del', case=False))\
            &(data['Variation'].str.contains('delins', case=False))),
            'mutation_type']= 'Deletion-Insertion'
    data.loc[(data['Variation'].str.contains('dup', case=False)), 'mutation_type']= 'Duplication'
    data.loc[(data['Variation'].str.contains('trunc', case=False)), 'mutation_type']= 'Truncation'
    data.loc[(data['Variation'].str.contains('fs', case=False)), 'mutation_type']= 'Frame Shift'
    data.loc[(data['Variation'].str.contains('splice', case=False)), 'mutation_type']= 'Splice'
    data.loc[(data['Variation'].str.contains('exon', case=False)), 'mutation_type']= 'Point Mutation'
    data.loc[((data['Variation'].str.contains('EGFR', case=False))\
            |(data['Variation'].str.contains('AR', case=True))\
            |(data['Variation'].str.contains('MYC-nick', case=True))\
            |(data['Variation'].str.contains('TGFBR1', case=True))\
            |(data['Variation'].str.contains('CASP8L', case=True))),
            'mutation_type']= 'Gene Subtype'
    data.loc[((data['Variation'].str.contains('Hypermethylation', case=False))\
            |(data['Variation'].str.contains('Epigenetic', case=False))),
             'mutation_type']= 'Epigenetic'
    data.loc[(data['mutation_type'].isin(major_types) == False),
            'mutation_type']= 'Others'

    # rearrange order of columns
    if 'Class' in data.columns:
        data = data[['ID', 'Gene', 'Variation', 'mutation_type', 'Class']]
    else:
        data = data[['ID', 'Gene', 'Variation', 'mutation_type']]

    return data
