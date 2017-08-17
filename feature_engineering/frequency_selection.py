import re
import numpy as np
import pandas as pd

from nltk import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

one_class_remove_list = ['bunkyo', 'commonest', 'commonplac', 'concret', 'consol',
                         'conspicu', 'credenc', 'damage—unlik', 'drew', 'enumer', 'logo',
                         'graduat','ibaraki', 'joshi', 'kaneda', 'kurumizaka', 'lesson',
                         'matsui', 'minami', 'minato', 'montreal', 'newyork', 'ontario',
                         'shirokanedai', 'sinai', 'taipei', 'wake', 'wise', 'yokohama']

def getRelativeDifference(a, b, freq_threshold=0.05):
    if ((b != 0) and (a >= freq_threshold)):
        return (a / b)
    else:
        return 0

def getAbsoluteDifference(a, b, freq_threshold=0.05):
    if ((a >= b ) and (a >= freq_threshold)):
        return (a - b)
    else:
        return 0

def decideOnDifference(a, b, freq_threshold, min_difference, mode='relative'):
    if mode == 'relative':
        difference = getRelativeDifference(a, b, freq_threshold=freq_threshold)
        return (difference >= min_difference)
    elif mode == 'absolute':
        difference = getAbsoluteDifference(a, b, freq_threshold=freq_threshold)
        return (difference >= min_difference)
    else:
        print('ERROR: Invalid mode')
        return

def getNClassWords(docs, doc_type='fraction_of_docs', mode='relative',
                    min_frequency=0.3, min_difference=1.5, print_result=True):
    '''
    This function looks at each word in each doc in the 'docs' and creates
    a list containing frequency of appearance ('apps') for each class. The list is
    re-order in descending order, and the fold difference between the adjacent pair
    of frequencies are compared. If the fold difference is above a certain threshold
    ('fold_threshold), the word is classified into a designated class of words.
    When a word is classified as n-class word, it means that the freqs of app of the
    word in n (number of) classes are X fold (fold_threshold) higher than those of
    other classes. freq_threshold is a cutoff freq of app to decide whether to
    include the word or not in the list.

    - Number of classes are assumed to be 9.
    - Use get_FoldDifference function to calculate fold difference

    INPUTS:
    ========
    frac_docs : DataFrame
        A list of lists containing fractions of docs a word appears in the class

    OUTPUTS:
    ========
    n_class_words : dictionary
        A dictionary whose keys are n-class_word labels. Values a lists of words
        in each of n classes of words
    '''
    min_freq = min_frequency
    min_d = min_difference
    mode = mode

    # Basic validation of input parameters
    if doc_type == 'fraction_of_docs':
        if ((min_freq < 0) or (min_freq > 1)):
            print('ERROR: freq_threshold must be between 0 and 1')
            return
        #else:
            #print('Doc Type: Fraction of Docs')
    if doc_type == 'per_doc_frequency':
        if (min_freq < 0):
            print('ERROR: freq_threshold must be above 0')
            return
        #else:
            #print('Doc Type: Per-doc Frequency')
    else:
        if not((doc_type == 'fraction_of_docs') or (doc_type == 'per_doc_frequency')):
            print('ERROR: Invalid Doc Type')
            return

    if not ((mode == 'relative') or (mode == 'absolute')):
        print('ERROR: Invalid Mode')

    if ((doc_type == 'fraction_of_docs') and (mode == 'relative')):
        if (min_freq*min_d) > 1:
            print('ERROR: Fold difference too high. Lower min_difference')
            return

    if ((doc_type == 'fraction_of_docs') and (mode == 'absolute')):
        if (min_freq + min_d) > 1:
            print('ERROR: Absolute difference too high. Lower min_difference')
            return

    ncw_labels = ['one_class_words', 'two_class_words', 'three_class_words',
                  'four_class_words', 'five_class_words', 'six_class_words',
                  'seven_class_words', 'eight_class_words','other_words']

    # Create a new dictionary to contain each n-class of words in list formats
    n_class_words = {}
    for i in range(9):
        n_class_words[ncw_labels[i]] = []

    # Get words for each n-class of words (might be a better way to do this?)
    for j, word in enumerate(docs.index):
        apps = np.array(docs.loc[word])
        apps[::-1].sort()
        if decideOnDifference(apps[0], apps[1],
                              freq_threshold=min_freq, min_difference=min_d, mode=mode):
            n_class_words[ncw_labels[0]].append(word)
        elif decideOnDifference(apps[1], apps[2],
                              freq_threshold=min_freq, min_difference=min_d, mode=mode):
            n_class_words[ncw_labels[1]].append(word)
        elif decideOnDifference(apps[2], apps[3],
                              freq_threshold=min_freq, min_difference=min_d, mode=mode):
            n_class_words[ncw_labels[2]].append(word)
        elif decideOnDifference(apps[3], apps[4],
                              freq_threshold=min_freq, min_difference=min_d, mode=mode):
            n_class_words[ncw_labels[3]].append(word)
        elif decideOnDifference(apps[4], apps[5],
                              freq_threshold=min_freq, min_difference=min_d, mode=mode):
            n_class_words[ncw_labels[4]].append(word)
        elif decideOnDifference(apps[5], apps[6],
                              freq_threshold=min_freq, min_difference=min_d, mode=mode):
            n_class_words[ncw_labels[5]].append(word)
        elif decideOnDifference(apps[6], apps[7],
                              freq_threshold=min_freq, min_difference=min_d, mode=mode):
            n_class_words[ncw_labels[6]].append(word)
        elif decideOnDifference(apps[7], apps[8],
                              freq_threshold=min_freq, min_difference=min_d, mode=mode):
            n_class_words[ncw_labels[7]].append(word)
        else:
            n_class_words[ncw_labels[8]].append(word)

    # Remove a list of words from one-class words
    n_class_words['one_class_words'] = [word for word in n_class_words['one_class_words'] if len(word) > 2]

    if print_result:
        print('======== n-class words extractions by %s differecne ========' % mode)
        print('Input Type: %s' % doc_type)
        print('Minimum Difference = %.2f' % min_d)
        print('Minimum Frequency = %.2f' % min_freq)
        total = 0
        for i in range(9):
            print('# of words in %s: %d' % (ncw_labels[i], len(n_class_words[ncw_labels[i]])))
            total += len(n_class_words[ncw_labels[i]])
        print('Total # of words: %d' % total)

    return n_class_words

def getNClassExclusiveWords(docs, min_frequency=0.2, print_result=True):
    '''
    This function looks at each word in each doc in the 'docs' and creates
    a list containing frequency of appearance ('apps') for each class. Then,
    the number of non-zero freqs in the list is counted. That number is used
    to classified the word as n-class word. In this case, n-class word is a
    word that appears only in n number of classes. freq_threshold parameter
    is a cutoff freq of app to decide whether to include the word or not in
    the list.

    - Number of classes are assumed to be 9
    - This function can be used for both 'fraction_of_docs' and 'perdoc_apps'
      types of inputs

    INPUTS:
    ========
    frac_docs : DataFrame
        A list of lists containing fractions of docs a word appears in the class

    OUTPUTS:
    ========
    n_class_words : dictionary
        A dictionary whose keys are n-class_word labels. Values a lists of words
        in each of n classes of words

    '''
    min_freq = min_frequency

    ncw_labels = ['one_class_words', 'two_class_words', 'three_class_words',
                  'four_class_words', 'five_class_words', 'six_class_words',
                  'seven_class_words', 'eight_class_words','other_words']

    # Create a new dictionary to contain each n-class of words in list formats
    n_class_words = {}
    for i in range(9):
        n_class_words[ncw_labels[i]] = []

    # Get words for each n-class of words
    for j, word in enumerate(docs.index):
        apps = np.array(docs.loc[word])
        num_nonzeros = np.count_nonzero(apps)
        if np.min(apps[np.nonzero(apps)]) >= min_freq:
            n_class_words[ncw_labels[(num_nonzeros-1)]].append(word)
        else:
            n_class_words[ncw_labels[8]].append(word)

    # Remove a list of words from one-class words
    one_class_remove_list = ['bunkyo', 'commonest', 'commonplac', 'concret', 'consol',
                             'conspicu', 'credenc', 'damage—unlik', 'drew', 'enumer', 'logo',
                             'graduat','ibaraki', 'joshi', 'kaneda', 'kurumizaka', 'lesson',
                             'matsui', 'minami', 'minato', 'montreal', 'newyork', 'ontario',
                             'shirokanedai', 'sinai', 'taipei', 'wake', 'wise', 'yokohama']
    n_class_words['one_class_words'] = [word for word in n_class_words['one_class_words'] if len(word) > 2]

    if print_result:
        print('======== n-class words extractions by exclusive appearances ========')
        print('Minimum Frequency = %f' % min_freq)
        total = 0
        for i in range(9):
            print('# of words in %s: %d' % (ncw_labels[i], len(n_class_words[ncw_labels[i]])))
            total += len(n_class_words[ncw_labels[i]])
        print('Total # of words: %d' % total)

    return n_class_words
