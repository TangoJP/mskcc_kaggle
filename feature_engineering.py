# Functions and Information for Engineering Features for MSKCC Competition
import numpy as np
import pandas as pd
import re
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# List of stop words to be exclude
stop_words = set(stopwords.words('english'))

# List of words too common to be kept
commoners = ['RT', 'PCR', 'RT-PCR', 'DNA', 'cDNA', 'RNA', 'mRNA', 'siRNA',
             'cell', 'cancer', 'CHIP', 'FISH', 'SDS-PAGE', 'UK', 'USA', 'GST',
             'shRNA', 'protein', 'basis', 'COHORT', 'OUTCOME', 'AIRWAY', 'EMSA',
             'GFP', 'SDS', 'PAGE', 'ANOVA', 'RIKEN',
             'qPCR', 'PBS', 'TBS', 'DTT', 'BSA', 'HSA', 'HCl', 'NCBI', 'PBST',
             'electrophoresis', 'hypothesis', 'hypothetical', 'analysis',
             'nonparametric', 'Malaysia', 'Asia', 'Indonesia', 'Pan-Asia',
             'Russia', 'Romania', 'Media', 'media',
             'australasia', 'tunisia']

#biowords = ['promoter', 'enhancer', 'neuron', 'marrow',
#            'loss-of-heterozygosity', 'loss of heterozygosity',
#            'progenitor', 'pluripotent']

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

def get_positive_and_negative_words(tokens):
    '''
    Get words that end with '-positive' and '-negative'.
    All the words will be converted to lower case.
    **This function retunrs a list and same words can appear multiple times.**

    INPUT:
    =====
    tokens : list
        list of tokenized words

    OUTPUT:
    =======
    pos_neg : list
        list of words that end with '-positive' and '-negative'
    '''
    pos_neg = [token.lower() for token in tokens \
           if (('-positive' in token.lower()) | ('-negative' in token.lower())) \
           if not (('false' in token.lower()) | ('true' in token.lower()))]

    # Remove GFP tag
    for i, word in enumerate(pos_neg):
        if 'egfp/eyfp' in word:
            pos_neg[i] = word.replace('egfp/eyfp-', '')
        elif 'egfp' in word:
            pos_neg[i] = word.replace('egfp-', '')
        elif 'gfp' in word:
            pos_neg[i] = word.replace('gfp'+"–", '') #This hyphen is somehow different from the other one

    pos_neg = [token for token in pos_neg \
               if not ((token == '-positive') | (token == '-negative'))]

    return pos_neg

def get_fusion_like_words(tokens, commoners=commoners):
    '''
    Get words that seem like a gene fusion name. This would also pick up words
    whose gene name has hyphens. Words like 'RT-PCR' can also be picked up, but
    some of them are (hopefully) removed since they're in the commoners list.
    All the words will be converted to lower case.
    **This function retunrs a list and same words can appear multiple times.**

    INPUT:
    ======
    tokens : list
        a list of tokenized words
    commoners : list
        a list of words that should be removed. Usually the one at the top of
        this file is used.

    OUTPUT:
    =======
    fusion_like : list
        a list of funsion-like words
    '''
    # Regex pattern to remove nucleotide sequences
    nts = r"[ACTGU]{4,}"

    # fusion-like pattern
    fusion_like_pattern = \
                r"([A-Z]{2,7})([0-9]{1,4})?-([A-Z]{0,7})([0-9]{1,4})?$"

    fusion_like = [word.lower() for word in tokens \
                   if word not in commoners \
                   if re.search(fusion_like_pattern, word) \
                   if not re.search(nts, word) \
                   if not re.search('gst-', word.lower())]

    # Remove GST-tagg
    for i, word in enumerate(fusion_like):
        if 'gst-' in word:
            fusion_like[i] = word.replace('gst-', '')

    return fusion_like

def get_race_words(tokens_white):
    '''
    Get words for race using the list inside the function

    INPUT:
    ======
    tokens_white : list
        A list of tokenized words, not containing non-alphanumerical characters

    OUTPUT:
    =======
    race_words : list
        A list of words describing race
    '''
    # words for races
    races = ['asian', 'hispanic', 'african', 'caucasian', 'native american',
             'indian', 'pacific islander', 'black', 'white', 'latino']

    race_words = [word.lower() for word in tokens_white if word.lower() in races]

    return races

def get_tissue_type_words(tokens_white):
    '''
    Get words for race using the list inside the function

    INPUT:
    ======
    tokens_white : list
        A list of tokenized words, not containing non-alphanumerical characters

    OUTPUT:
    =======
    organ_words : list
        A list of words describing race
    '''
    # list of organ-related words
    organs = ['brain', 'liver','skin', 'stomach', 'gastric', 'intestine',
              'intestinal', 'colon', 'rectum', 'rectal', 'parathyroid',
              'prostate', 'breast', 'ovary', 'ovarian', 'kidney', 'renal',
              'thyroid', 'esophogus', 'esophogal', 'bone', 'spinal', 'heart',
              'pancreas', 'pancreatic', 'adrenal', 'gland', 'pituitary',
              'spleen', 'splenic', 'bladder', 'gallbladder', 'lung', 'cardiac',
              'cervix', 'cervical', 'skeletal']
    # Get words for organs
    organ_words = [word.lower() for word in tokens_white \
                   if word.lower() in organs]

    # Get tumor type words, remove plural s at the end
    tumor_pattern = r"[A-Za-z]+omas?$"
    tumor_words = [word.lower() for word in tokens_white \
                   if re.search(tumor_pattern, word)]
    tumor_words = [re.sub(r's$', '', word) for word in tumor_words]

    # Get cell type words
    celltype_pattern1 = r"[A-Za-z]+cytes?$"
    celltype_pattern2 = r"\w+blasts?$"
    celltype_words = [word.lower() for word in tokens_white \
                      if re.search(celltype_pattern1, word)]
    celltype_words += [word.lower() for word in tokens_white \
                       if re.search(celltype_pattern2, word)]
    celltype_words = [re.sub(r'\W', '', word) for word in celltype_words]
    celltype_words = [re.sub(r's$', '', word) for word in celltype_words]

    tissue_type_words = organ_words + tumor_words + celltype_words

    return tissue_type_words

def get_drug_words(tokens_white):
    '''
    Get durg names used for (mostly cancer) therapies that end with one of the
    followings: -mib, -nib, and -mab

    INPUT:
    ======
    tokens_white : list
        A list of tokenized words, not containing non-alphanumerical characters

    OUTPUT:
    drug_words : list
        A list of words containing the word endings described above
    '''
    drug_pattern = r"\w+nib$|\w+mib$|\w+mab$"
    drug_words = [word.lower() for word in tokens_white \
                    if re.search(drug_pattern, word)]

    return drug_words

def get_gene_like_words(tokens_white, commoners=commoners):
    '''
    Get words that have gene-like names.

    INPUT:
    ======
    tokens_white : list
        A list of tokenized words, not containing non-alphanumerical characters
    commoners : list
        A list containing common words to be removed

    OUTPUT:
    =======

    '''
    # Remove nucleotide sequences
    nts = r"[ACTGU]{4,}"

    tokens_white = [word for word in tokens_white \
                    if not word in stop_words \
                    if not word in commoners \
                    if not re.search(nts, word)]

    gene_like_pattern1 = r"^[A-Z]{2,7}-?[0-9]{0,4}$"    # e.g. NUP100
    gene_like_pattern2 = r"^p[0-9]{1,3}$"               # e.g p53
    gene_like_words = [word.lower() for word in tokens_white \
                       if re.match(gene_like_pattern1, word)]
    gene_like_words += [word.lower() for word in tokens_white \
                       if re.match(gene_like_pattern2, word)]

    return gene_like_words

def get_protein_words(tokens_white, commoners=commoners):
    '''
    Get words related to proteins, which include words obtained by
    get_gene_like_words function above. Other words include enzymes ending with
    -ase(S) and post-translational modifications (PTMs)

    INPUT:
    ======
    tokens_white : list
        A list of tokenized words, not containing non-alphanumerical characters

    OUTPUT:
    =======
    protein_words : list
        A list of words related to proteins
    '''
    # Get gene-like words
    gene_like_words = get_gene_like_words(tokens_white, commoners=commoners)

    # Get enzyme words
    enzyme_pattern= r"\w+ases?$"
    enzyme_words = [word.lower() for word in tokens_white \
                     if re.search(enzyme_pattern, word)]
    enzyme_words = [re.sub(r's$', '', word) for word in enzyme_words]

    # Get PTM words
    ptm_pattern = r"^glyco|^phospho|^ubiquityl|^ubiquitinat|\
                      ^acetyl|^methyl|^deamin|^oxydat"
    ptm_words = [word for word in tokens_white \
                 if re.search(ptm_pattern, word) \
                 if word not in commoners]
    ptm_words = [re.sub(r's$', '', word) for word in ptm_words]

    # Combined all
    protein_words = gene_like_words + enzyme_words + ptm_words

    return protein_words

def get_biomedical_words(tokens_white):
    '''
    Get other biomedical terms not covered in the other functions, such as ones
    containing suffix '-ia(s)' and other prefixes.

    INPUT:
    ======
    tokens_white : list
        A list of tokenized words, not containing non-alphanumerical characters

    OUTPUT:
    =======
    biomedical_words : list
        A list of biomedical words
    '''
    # Get words ending with '-ia(s)'
    ias_to_remove = ['bolognia', 'lithuania', 'northumbria', 'pan‑asia',
        'westphalia', 'yugoslavia', 'mejia', 'damia', 'sylvia', 'bhatia',
        'carpintenia', 'sisodia', 'bia', 'arabia', 'catalonia', 'mdia',
        'cynthia', 'xia', 'victoria', 'tunisia', 'oceania',
        'farugia', 'australasia', 'cassia', 'arteria', 'casaccia', 'youjia',
        'walia', 'cornelia', 'sarkaria', 'savoia', 'rsalgia',
        'macia', 'algeria', 'rangatia', 'sequoia', 'anodontia', 'bonavia',
        'sanitaria', 'center–sophia', 'mangia', 'rozovskaia',
        'georgia', 'indústria', 'austria', 'sotoodehnia', '6australia',
        'biovia', 'virginia', 'valsesia', 'gallia', 'valencia', 'perugia',
        'silvia', 'pennsylvania', 'philadelphia', 'tartaglia', 'behzadnia',
        'wikipedia', 'garcia', 'sonia', 'hafsia', 'tolia', 'pavia',
        'slovakia', 'elia', 'mathia', 'l6czechosiovakia', 'iglesia', 'giaccia',
        'belandia', 'baldia', 'materia', 'tulia', 'eurasia',
        'santamaria', 'italia', 'academia', 'sushia', 'soravia', 'gloria',
        'bagrodia', 'india', 'caria', 'ethiopia', 'ansonia', 'maria',
        'mtartaglia', 'dahia', 'australia', 'eugenia', 'catania', 'luria',
        'coria', 'titia', 'mattia', 'consortia', '9philadelphia',
        'ghia', 'sardinia', 'gaia', 'capinteria', 'terapia', 'faria',
        'colombia', 'farrugia', 'via', 'tria', 'nigeria', '1ia', 'emilia',
        'vallania', 'california', 'sophia', 'farma´cia', 'patricia',
        'santarpia', 'rumania', 'cristália', 'bosottia', 'mongia', 'gradia',
        '177030criteria', 'aria', 'scotia', 'slovenia', 'lucia', 'bavaria',
        'griffonia', 'nia', 'alia', 'jia', 'tasmania', 'scalia', 'natalia',
        'maia', 'tobia', 'garcdia', 'estonia', 'columbia', 'candia', 'criteria',
        'monteia', 'attia', 'provia', 'sanabria', 'matthia', 'sibilia',
        'iberia', 'sebia', 'beria', 'ilia', 'tapia']

    pattern1 = r"[A-Za-z]+[sm]?ias?$"
    words_with_sias = [word.lower() for word in tokens_white \
                     if re.search(pattern1, word)\
                     if word not in commoners \
                     if word.lower() not in ias_to_remove]
    words_with_sias = [re.sub(r's$', '', word) for word in words_with_sias]
    words_with_sias = [re.sub(r'ae', 'e', word) for word in words_with_sias]
    words_with_sias = [re.sub(r'\W', '', word) for word in words_with_sias]

    # Get words ending with '-ic' and '-is'
    ics_to_remove = ['antarctic', 'electric', 'specific', 'nonspecific',
                     'this', 'volumetric', 'thesis', 'terrific', 'horrific',
                     'graphic', 'mechanic', 'mechanistic', 'basis', 'analysis']

    pattern2 = r"[A-Za-z]+i[sc]$"
    words_with_is_ic = [word.lower() for word in tokens_white \
                         if re.search(pattern2, word) \
                         if word not in commoners \
                         if word.lower() not in ics_to_remove]

    # Get words with variaus prefixes
    prefix_pattern = r"^epiderm|^endothel|^onco|^hepato|^haemato|^osteo|\
                        ^neuro|^cholecyst|^cyst|^encephal|^erythr|^gastr|\
                        ^hist|^karyo|^kerat|^lymph|myel|^necr|^nephr|^sarco|\
                        ^terato|^thorac|^trache|^vasculo|^hyper|^hypo"
    words_various_prefixes = [word.lower() for word in tokens_white \
                              if re.search(prefix_pattern, word)]

    # combine them
    biomedical_words = words_with_sias + words_with_is_ic \
                       + words_various_prefixes

    return biomedical_words

def get_mutation_type_words(tokens_white):
    '''
    Get words contained in the mutation_patterns list included in the function.
    The mutation_pattern list contains mutation type words that are the sames
    as the ones to characterized the mutation variation in the 'variants' text.
    **This function retunrs a list and same words can appear multiple times.**

    INPUT:
    ======
    tokens_white : list
        A list of tokenized words, not containing non-alphanumerical characters

    OUTPUT:
    =======
    mutation_type_words : list
        A list of mutation type words
    '''
    mutation_patterns = ['truncation', 'deletion', 'promoter',
                     'amplification', 'epigenetic', 'frame',
                     'overexpression', 'duplication', 'insertion',
                     'subtype', 'fusion', 'splice', 'wildtype']

    mutation_type_words = [word.lower() for word in tokens_white \
                           if word.lower() in mutation_patterns]

    return mutation_type_words

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
