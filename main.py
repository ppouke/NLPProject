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





#import testC
path = os.path.dirname(os.path.abspath(__file__))
files = "testCorpus.txt"
corpus0 = PlaintextCorpusReader(path, files)
testCorpus  = nltk.Text(corpus0.words())


#bnc remove stopwords
es = set(stopwords.words('english'))
content = [w for w in bncwords if w[0].lower() not in es]

# tgm = nltk.collocations.TrigramAssocMeasures()
# finder = TrigramCollocationFinder.from_words(content)
#
# finder.apply_freq_filter(3)


def findMutualInformation(corpus, testWords, corpus_size, span):

    tagless = [x[0] for x in content]
    fd = FreqDist(tagless)

    metaphorWords = {}
    for t in testWords:
        metaphorWords[t] = []

    for word in testWords:
        finder = BigramCollocationFinder.from_words(corpus)
        fdfilter = lambda *w: word not in w[0]
        tagfilter = lambda w1, w2: "VERB" not in w2[1] and "ADJ" not in w2[1] and "ADV" not in w2[1]
        finder.apply_ngram_filter(fdfilter)
        finder.apply_ngram_filter(tagfilter)

        A = fd[word]
        for c in finder.ngram_fd.items():
            col = c[0][1][0]
            B = fd[col]
            AB = c[1]
            mi = mutInf(A, B, AB, corpus_size, span)
            if mi >= 3:
                metaphorWords[word].append(col)

            #mi_like score
            #scored = finder.score_ngrams(bigram_measures.mi_like)
            #print(scored)
    return metaphorWords

#1. Find metaphors for words in BNC
testWords = ["woman", "use", "dream", "body"]
size_corpus = len(bncwords)
#mets = findMutualInformation(content, testWords, size_corpus, 4)
#print(mets)

#2. test Metaphor finding w/ testCorpus

headwords = {}
f = open("testCorpus.txt", "r")
lines = f.readlines()
for count, line in enumerate(lines):


    words = line.split()
    headwordIndex = int(words[-1].split("@")[1])
    headword = re.sub(r'[^a-zA-Z ]', '', str(count) + " : " + words[headwordIndex -1])
    headwords[headword] = []
    for i in [-3, -2, 0, 1]:
        headwords[headword].append(words[headwordIndex + i])
        

print(headwords)
