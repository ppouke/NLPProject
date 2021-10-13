import nltk
from nltk.collocations import *
from nltk.corpus import PlaintextCorpusReader
from nltk.corpus import stopwords
from nltk import FreqDist, bigrams
from nltk import Text
import numpy as np
from math import log
import os





#import brown corpus
from nltk.corpus import brown

#import testC
path = os.path.dirname(os.path.abspath(__file__))
files = ".*\.txt"
corpus0 = PlaintextCorpusReader(path, files)
testCorpus  = nltk.Text(corpus0.words())



#brown remove stopwords

es = stopwords.words('english')
content = [w for w in brown.words() if w.lower() not in es]

# tgm = nltk.collocations.TrigramAssocMeasures()
# finder = TrigramCollocationFinder.from_words(content)
#
# finder.apply_freq_filter(3)



#MutInf for mann
word = "man"

fd = FreqDist(content)
A = fd['man']








def mutInf(A, B, AB, sizeCorpus, span):
    # A = frequency of node word
    # B = frequency of collocate
    # AB = frequency of collocate near the node word
    # span = span of words around node word
    return log( (AB * sizeCorpus)/(A*B*span) ) / 0.30103
