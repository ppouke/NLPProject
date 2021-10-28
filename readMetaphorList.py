import nltk
from nltk.collocations import *
from nltk.corpus import PlaintextCorpusReader
from nltk.corpus import stopwords
from nltk import FreqDist, bigrams
from nltk import Text
import numpy as np
from math import log
import os
import re
import pickle
from os.path import exists
import copy



def readMetList():
    es = set(stopwords.words('english'))

    f = open("metaphorList.txt", "r")
    lines = f.readlines()


    path = os.path.dirname(os.path.abspath(__file__))
    files = "metaphorList.txt"
    corpus1 = PlaintextCorpusReader(path, files)
    mList  = corpus1.words()

    mListNS = [w for w in mList if w not in es and w.isalpha()]

    metfinder = BigramCollocationFinder.from_words(mListNS)
    punctfilter = lambda *w: len(w[0]) <= 1 or len(w[1]) <= 1
    type3Filter = lambda w1, w2: 'JJ' not in nltk.pos_tag([w1])[0][1] or "NN" not in nltk.pos_tag([w2])[0][1]

    metfinder.apply_ngram_filter(punctfilter)
    metfinder.apply_ngram_filter(type3Filter)

    type3mets = metfinder.ngram_fd.items()

    deadOrAlive = []

    for count, line in enumerate(lines):
        tokens = line.split()
        if "L" in tokens[0]:
            deadOrAlive.append(True)
        else:
            deadOrAlive.append(False)

    return deadOrAlive,type3mets, lines
