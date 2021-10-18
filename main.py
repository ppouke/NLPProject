import nltk
from nltk.collocations import *
from nltk.corpus import PlaintextCorpusReader
from nltk.corpus import stopwords
from nltk import FreqDist, bigrams
from nltk import Text
import numpy as np
from math import log
import os

from nltk.corpus.reader.bnc import BNCCorpusReader
from nltk.collocations import BigramAssocMeasures, BigramCollocationFinder

bnc_reader = BNCCorpusReader(root="download\Texts", fileids=r'[A]/\w*/\w*\.xml')

list_of_fileids = ['A/A0/A00.xml', 'A/A0/A01.xml', 'A/A0/A02.xml', 'A/A0/A03.xml', 'A/A0/A04.xml', 'A/A0/A05.xml']
bigram_measures = BigramAssocMeasures()
bncwords = bnc_reader.words(fileids=list_of_fileids)
#scored = finder.score_ngrams(bigram_measures.raw_freq)

#print(scored)

#import testC
path = os.path.dirname(os.path.abspath(__file__))
files = ".*\.txt"
corpus0 = PlaintextCorpusReader(path, files)
testCorpus  = nltk.Text(corpus0.words())


#brown remove stopwords

es = set(stopwords.words('english'))
content = [w for w in bncwords if w.lower() not in es]

# tgm = nltk.collocations.TrigramAssocMeasures()
# finder = TrigramCollocationFinder.from_words(content)
#
# finder.apply_freq_filter(3)

fd = FreqDist(content)

testWords = ["woman", "use", "dream", "body"]

for word in testWords:
    finder = BigramCollocationFinder.from_words(content)
    fdfilter = lambda *w: word not in w
    finder.apply_ngram_filter(fdfilter)
    scored = finder.score_ngrams(bigram_measures.mi_like)
    print(fd[word])
    #print(scored)

def mutInf(A, B, AB, sizeCorpus, span):
    # A = frequency of node word
    # B = frequency of collocate
    # AB = frequency of collocate near the node word
    # span = span of words around node word
    return log( (AB * sizeCorpus)/(A*B*span) ) / 0.30103
