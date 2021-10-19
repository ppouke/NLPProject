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
    if A == 0 or B == 0 or AB == 0:
        return 0
    else:
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
tagless = [x[0] for x in content]
fd = FreqDist(tagless)

# tgm = nltk.collocations.TrigramAssocMeasures()
# finder = TrigramCollocationFinder.from_words(content)
#
# finder.apply_freq_filter(3)


def findMutualInformation(corpus, testWords, corpus_size, span):


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
groundTruth = []
f = open("testCorpus.txt", "r")
lines = f.readlines()
for count, line in enumerate(lines):


    words = line.split()
    headwordIndex = int(words[-1].split("@")[1]) - 1
    groundTruth.append(line[-2])
    headword = re.sub(r'[^a-zA-Z ]', '', words[headwordIndex]).strip()
    headword += " " + str(count + 1)
    headwords[headword] = []

    for j in range(-1, 2, 2):
        i = j
        foundwords = 0
        while foundwords < 2:
            if headwordIndex + i >= len(words) - 1 or headwordIndex + i < 0:
                break
            if words[headwordIndex+i] not in es and len(re.sub(r'[^a-zA-Z ]', '',words[headwordIndex+i]).strip()) > 1 :
                headwords[headword].append(words[headwordIndex+i])
                foundwords += 1
            i += j

#print(headwords)
#print(groundTruth)

#calculate MI usinb BNC
sentMiScore = []
finder = BigramCollocationFinder.from_words(tagless)
index = 0
for h,adjacent in headwords.items():
    index +=1
    headword = h.split()[0]
    A = fd[headword]
    sumMi = 0
    numAdj = 0
    for col in adjacent:
        B = fd[col]
        AB = finder.ngram_fd[(headword, col)]
        mi = mutInf(A, B, AB, size_corpus, 4)

        numAdj += 1
        sumMi += mi


    if numAdj == 0:
        print(h)
        avgMi = 0
    else:
        avgMi = sumMi/numAdj
    sentMiScore.append(avgMi)

#calculate accuracy(= correct/all)
numAll = 0
numCorrect = 0
for count, gT in enumerate(groundTruth):
    if gT == "s":
        continue
    else:
        numAll += 1
        scoreOver3 = sentMiScore[count] >= 3
        isMetaphor = gT == "y"
        if scoreOver3 != isMetaphor:
            numCorrect += 1

accuracy = numCorrect/numAll
print(accuracy)


#3. Import wordnet
#using nltk wordnet (change to provalis?)
from nltk.corpus import wordnet as wn
def findWordCategories(word, POS):
    #POS = e.g. wn.VERB
    for synset in wn.synsets(word, pos=POS):
        #print(str(synset) + " : " + synset.lexname())


#4.
