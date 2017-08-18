import numpy as np
import pandas as pd
import re
from collections import Counter
from nltk import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

def process_text1(text, print_on=False):
    '''
    Process the original text. Tokenize into words first,
    and then remove stop words and numbers

    INPUT:
    ======
    text : str
        A string containing a writing to be analyzed

    OUTPUT:
    =======
    words : list
        A list of tokenized words

    '''
    # Tokenize the text
    word_tokens = word_tokenize(text)

    # Remove some unwanted words (hyphen excluded), and numbers
    remove_list = ['.',',','(',')','[',']','=','+','>','<',':',';','%']
    word_tokens = [word for word in word_tokens if word not in remove_list]
    word_tokens = [word for word in word_tokens if (word.isnumeric() == False)]

    # Remove Stop words
    stop_words = set(stopwords.words('english'))
    words = [w for w in word_tokens if not w in stop_words]

    # print if print_on=True
    if print_on:
        print('Length Before removing stop words %d' % len(word_tokens))
        print('Length After removing stop words %d' % len(words))

    return words

def replace_with_whitespace(text, hyphens='off'):
    '''
    Replace non-alphanumerical characters with a white space. When hyphen='on',
    hyphens will be also removed.

    INPUT:
    ======
    text : str
        a text in string format to be processed
    OUTPUT:
    =======
    text_white : str
        a processed text
    '''
    text_white = text.encode().decode()  # copy a string?!
    text_white = text_white.replace('"', ' ')
    text_white = text_white.replace('.', ' ')
    text_white = text_white.replace(',', ' ')
    text_white = text_white.replace(':', ' ')
    text_white = text_white.replace(';', ' ')
    text_white = text_white.replace('@', ' ')
    text_white = text_white.replace('<', ' ')
    text_white = text_white.replace('>', ' ')
    text_white = text_white.replace('/', ' ')
    text_white = text_white.replace('\'', ' ')
    text_white = text_white.replace('_', ' ')
    text_white = text_white.replace('=', ' ')
    text_white = text_white.replace('\n', ' ')
    text_white = text_white.replace('\\n', ' ')
    text_white = text_white.replace('\'', ' ')
    text_white = re.sub(' +',' ', text_white)
    text_white = text_white.replace('\'', ' ')
    text_white = text_white.replace('(', ' ')
    text_white = text_white.replace(')', ' ')
    text_white = text_white.replace('[', ' ')
    text_white = text_white.replace(']', ' ')
    text_white = text_white.replace('{', ' ')
    text_white = text_white.replace('}', ' ')
    if hyphens == 'on':
        text_white = text_white.replace('-', ' ')

    return text_white

def llwords2lstrs(list_of_listsOfWords):
    '''
    Convert a list of lists of words into a lis of strings.

    '''
    list_of_strings = []
    for j, doc in enumerate(list_of_listsOfWords):
        text = ' '.join(doc)
        list_of_strings.append(text)

    return list_of_strings
