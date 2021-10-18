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


def mutInf(A, B, AB, sizeCorpus, span):
    # A = frequency of node word
    # B = frequency of collocate
    # AB = frequency of collocate near the node word
    # span = span of words around node word
    return log( (AB * sizeCorpus)/(A*B*span) ) / 0.30103


bnc_reader = BNCCorpusReader(root="download\Texts", fileids=r'[A]/\w*/\w*\.xml')

list_of_fileids = ['A/A0/A00.xml', 'A/A0/A01.xml', 'A/A0/A02.xml', 'A/A0/A03.xml', 'A/A0/A04.xml', 'A/A0/A05.xml']
bigram_measures = BigramAssocMeasures()
bncwords = bnc_reader.tagged_words(fileids=list_of_fileids)
#scored = finder.score_ngrams(bigram_measures.raw_freq)


size_corpus = len(bncwords)
print(size_corpus)

#import testC
path = os.path.dirname(os.path.abspath(__file__))
files = ".*\.txt"
corpus0 = PlaintextCorpusReader(path, files)
testCorpus  = nltk.Text(corpus0.words())


#brown remove stopwords

es = set(stopwords.words('english'))
content = [w for w in bncwords if w[0].lower() not in es]

# tgm = nltk.collocations.TrigramAssocMeasures()
# finder = TrigramCollocationFinder.from_words(content)
#
# finder.apply_freq_filter(3)

tagless = [x[0] for x in content]
fd = FreqDist(tagless)

testWords = ["woman", "use", "dream", "body"]

for word in testWords:
    finder = BigramCollocationFinder.from_words(content)
    fdfilter = lambda *w: word not in w[0]
    tagfilter = lambda w1, w2: "VERB" not in w2[1] and "ADJ" not in w2[1] and "ADV" not in w2[1]


    finder.apply_ngram_filter(fdfilter)
    finder.apply_ngram_filter(tagfilter)


    A = fd[word]

    for c in finder.ngram_fd.items():
        col = c[0][1][0]

        B = fd[col]

        AB = c[1]

        mi = mutInf(A, B, AB, size_corpus, 2)

        print(mi)

    #mi_like score
    #scored = finder.score_ngrams(bigram_measures.mi_like)
    #print(scored)
