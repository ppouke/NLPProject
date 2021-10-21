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

import wdLoader as wl

from nltk.corpus.reader.bnc import BNCCorpusReader
from nltk.corpus import reuters



from nltk.collocations import BigramAssocMeasures, BigramCollocationFinder
from nltk.corpus import wordnet as wn





bnc_reader = BNCCorpusReader(root="download\Texts", fileids=r'[A]/\w*/\w*\.xml')
list_of_fileids = ['A/A0/A00.xml', 'A/A0/A01.xml', 'A/A0/A02.xml', 'A/A0/A03.xml', 'A/A0/A04.xml', 'A/A0/A05.xml']
bigram_measures = BigramAssocMeasures()
bncwords = bnc_reader.tagged_words()
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






def mutInf(A, B, AB, sizeCorpus, span):
    # A = frequency of node word
    # B = frequency of collocate
    # AB = frequency of collocate near the node word
    # span = span of words around node word
    if A == 0 or B == 0 or AB == 0:
        return 0
    else:
        return log( (AB * sizeCorpus)/(A*B*span) ) / 0.30103



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

def parseTestC():
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
def calAvgMi():
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
    return accuracy



#3. Import wordnet
#using nltk wordnet (change to provalis?)

def findWordCategories(word, POS):
    #POS is e.g. wn.VERB
    sets = [word]
    for synset in wn.synsets(word, pos=POS):
        sets.append(synset)

    return sets

#4.
def idSentence(sentence):
    tokens = nltk.word_tokenize(sentence)
    tagged = nltk.pos_tag(tokens)
    words = []

    #take first occuring Noun and adjective/adverb in sentence (remove break to get all)
    for w in tagged:
        if "NN" in w[1]:
            words.append(w)
            break
    for w in tagged:
        if "JJ" in w[1]:
            words.append(w)
            #print(w[0])
            break

    return words


def getWordCategories(sentence):
    nounSyns = None
    adjSyns = None
    for w in idSentence(sentence):
        if "NN" in w[1]:
            nounSyns = findWordCategories(w[0], wn.NOUN)
        if "JJ" in w[1]:
            adjSyns = findWordCategories(w[0], wn.ADJ)
    return (nounSyns, adjSyns)




cats = getWordCategories("this is a dark world")

def findIfMetaphor(categories):
    if len(categories[1]) == 2:
        return False
    if categories[0] == None:
        return None
    elif len(categories[0]) == 1:
        return None

    noun = categories[0][0]
    adj = categories[1][0]

    finder = BigramCollocationFinder.from_words(content)
    fdfilter = lambda *w: adj not in w[0]
    tagfilter = lambda w1, w2: 'SUBST' not in w2[1]
    punctfilter = lambda *w: len(w) <= 1
    finder.apply_ngram_filter(punctfilter)
    finder.apply_ngram_filter(fdfilter)
    finder.apply_ngram_filter(tagfilter)



    A = fd[adj]

    S = []
    for c in finder.ngram_fd.items():
        col = c[0][1][0]
        B = fd[col]

        AB = c[1]

        mi = mutInf(A, B, AB, size_corpus, 4)
        if mi >= 3:
            S.append((c, mi))

    S1all = []
    #filter away abstract
    for n in S:
        nC = findWordCategories(n[0][0][1][0], wn.NOUN)
        if len(nC) > 1:
            h1 = nC[1].hypernym_paths()[0][1]
            if "physical" in str(h1):
                S1all.append(n)

        else:
            return None



    #keep 3 with higest mutual info
    S1 = []
    if len(S1all) > 3:
        for i in range(3):
            S1.append(max(S1all,key=lambda item:item[1]))
            S1all.remove(max(S1all,key=lambda item:item[1]))
    else:
        S1 = S1all
    #print(S1)

    n = wn.synsets(noun)[0]
    #Wu-Palmer method


    # for e in S1:
    #     s = wn.synsets(e[0][0][1][0])[0]
    #
    #     similarity = s.wup_similarity(n)
    #
    #     if similarity > 0.4:
    #         return False
    #
    # return True

    #Word domain method
    wloader = wl.WordNetDomains(os.getcwd())
    nDomains = wloader.get_domains(noun)

    for e in S1:
        colDomains = wloader.get_domains(e[0][0][1][0])
        for d in colDomains:
            if d in nDomains:
                return False

    return True
#print(findIfMetaphor(cats))


#5. test with half the dataset.

parseTestC()


def testCorpusTest():
    mGuess = []
    for count, line in enumerate(lines):
        tokens = line.split()
        del tokens[-1]
        sent = ""
        for t in tokens:
            sent += " " + t
        cat = getWordCategories(sent)
        if cat[1] is not None:
            #print(cat[1][0])
            mGuess.append(str(count) + " : " + str(findIfMetaphor(cat)))
        else:
            mGuess.append(str(count) + ": None")

    #calculate accuracy(= correct/all)
    print(mGuess)
    numAll = 0
    numCorrect = 0
    numTrue = 0
    for count, gT in enumerate(groundTruth):
        if gT == "s" or "None" in mGuess[count] :
            continue
        else:
            numAll += 1
            predictedMetaphor = "True" in mGuess[count]
            isMetaphor = gT == "y"
            if predictedMetaphor == isMetaphor:
                numCorrect += 1
            if predictedMetaphor:
                numTrue += 1
    accuracy = numCorrect/numAll
    print(accuracy)
    print(numAll)
    print(numTrue)



testCorpusTest()
