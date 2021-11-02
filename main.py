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

import readMetaphorList as rml
import wdLoader as wl

from nltk.corpus.reader.bnc import BNCCorpusReader
from nltk.corpus import reuters
from nltk.corpus import brown



from nltk.collocations import BigramAssocMeasures, BigramCollocationFinder
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer



#import stopword list
es = set(stopwords.words('english'))
content = []
reutcontent = []
#import stopworded list if exists
if os.path.exists('stopworded.pkl'):
    print("bnc exists")
    with open("stopworded.pkl", "rb") as file:
        content = pickle.load(file)
else:
    bnc_reader = BNCCorpusReader(root="download\Texts", fileids=r'[A-B]/\w*/\w*\.xml')
    list_of_fileids = ['A/A0/A00.xml', 'A/A0/A01.xml', 'A/A0/A02.xml', 'A/A0/A03.xml', 'A/A0/A04.xml', 'A/A0/A05.xml']
    bigram_measures = BigramAssocMeasures()
    print("creating tagged word list")
    #bncwords = bnc_reader.tagged_words()

    bncwords = bnc_reader.words()

    size_corpus = len(bncwords)

    print("removing stopwords")
    #bnc remove stopwords
    content = [w for w in bncwords if w not in es and w.isalpha()]

    print("Done!")
    print(content[0])
    print(len(content))

    with open('stopworded.pkl', 'wb') as f:
        pickle.dump(content, f)

    print("Saved")

if os.path.exists("stopwordedReut.pkl"):
    print("reut exists")
    with open("stopwordedReut.pkl", "rb") as file:
        reutcontent = pickle.load(file)
else:
    reutwords = reuters.words()
    reutcontent = [w for w in reutwords if w not in es and w.isalpha()]
    with open('stopwordedReut.pkl', 'wb') as f:
        pickle.dump(reutcontent, f)

    print("Saved2")




path = os.path.dirname(os.path.abspath(__file__))
files = "testCorpus.txt"
corpus0 = PlaintextCorpusReader(path, files)
testCorpus  = corpus0.words()


"""_________________________"""
"""Change to False to use Reuters Corpus"""

usingReut = False
usingBNC = False


"""_________________________"""

size_corpus = 96263399
if usingReut:
    print("Using Reuters Corpus")
    content = reutcontent
    size_corpus = len(content)
    tagless = content

else:
    print("Using BNC Corpus")
    #tagless = [x[0] for x in content]
    tagless = content




print("loaded corpuses, creating tagless...")

fd = FreqDist(tagless)
print("Created tagless")




def mutInf(A, B, AB, sizeCorpus, span):
    # A = frequency of node word
    # B = frequency of collocate
    # AB = frequency of collocate near the node word
    # span = span of words around node word
    if A == 0 or B == 0 or AB == 0:
        #print("{},{},{}".format(A,B,AB))
        return 0
    else:
        return log( (AB * sizeCorpus)/(A*B*span) ) / 0.30103





def findMutualInformation(corpus, testWords, corpus_size, span):


    metaphorWords = {}
    for t in testWords:
        metaphorWords[t] = []


    miScores = []
    for word in testWords:
        finder = BigramCollocationFinder.from_words(corpus)
        fdfilter = lambda *w: word not in w[0]
        if usingBNC:
            tagfilter = lambda w1, w2: 'VB' not in w2[1] and 'JJ' not in w2[1] and 'RB' not in w2[1]
        else:
            tagfilter = lambda *w: "VB" not in nltk.pos_tag([w[1]])[0][1] and "JJ" not in nltk.pos_tag([w[1]])[0][1] and "RB" not in nltk.pos_tag([w[1]])[0][1]

        finder.apply_freq_filter(3)
        finder.apply_ngram_filter(fdfilter)
        finder.apply_ngram_filter(tagfilter)

        #if  word == "dream":
            #print(finder.ngram_fd.items())
        A = fd[word]

        for c in finder.ngram_fd.items():
            if usingBNC:
                col = c[0][1][0]
            else:
                col = c[0][1]
            #print(col)
            B = fd[col]

            AB = c[1]
            mi = mutInf(A, B, AB, corpus_size, span)
            miScores.append(mi)
            if mi >= miThresh:
                metaphorWords[word].append(col)

            #mi_like score
            #scored = finder.score_ngrams(bigram_measures.mi_like)
            #print(scored)

    print("avg mi = " + str(np.average(miScores)))
    return metaphorWords



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






def findIfMetaphor(categories, WuPalmer=True):

    if len(categories[1]) == 2:
        #print("only one synonym for adjective: {}".format(categories[1][0]))
        return False
    if categories[0] == None:
        return None
    if len(categories[0]) == 1:
        #print("noun not found in synonyms")
        return None

    noun = categories[0][0]
    adj = categories[1][0]



    #print("copying finder...")
    finder = copy.deepcopy(global_finder)
    #print("Done!")


    # print(global_finder)
    # print(finder)

    fdfilter = lambda *w: adj != w[0]
    if usingBNC:
        tagfilter = lambda w1, w2: 'NN' not in w2[1]
    else:
        tagfilter = lambda *w: 'NN' not in nltk.pos_tag([w[1]])[0][1]
    punctfilter = lambda *w: len(w[1]) <= 1
    #finder.apply_ngram_filter(punctfilter)
    #finder.apply_freq_filter(3)
    finder.apply_ngram_filter(fdfilter)
    finder.apply_ngram_filter(tagfilter)






    A = fd[adj]

    S = []
    for c in finder.ngram_fd.items():
        if usingBNC:
            col = c[0][1][0]
        else:
            col = c[0][1]

        B = fd[col]
        AB = c[1]

        mi = mutInf(A, B, AB, size_corpus, 4)
        if mi >= miThresh:
            S.append((c, mi))




    #filter away abstract
    S1all = []
    for n in S:
        if usingBNC:
            nC = findWordCategories(n[0][0][1][0], wn.NOUN)
        else:
            nC = findWordCategories(n[0][0][1], wn.NOUN)



        if len(nC) > 1:
            try:
                h1 = nC[1].hypernym_paths()[0][1]
            except:
                continue
            if "physical" in str(h1):
                S1all.append(n)

        else:
            continue



    if len(S1all) == 0:
        return None
    #keep 3 with higest mutual info
    S1 = []
    if len(S1all) > 3:
        for i in range(3):
            S1.append(max(S1all,key=lambda item:item[1]))
            S1all.remove(max(S1all,key=lambda item:item[1]))
    else:
        S1 = S1all


    n = wn.synsets(noun)[0]


    if WuPalmer:
        #Wu-Palmer method
        for e in S1:
            if usingBNC:
                s = wn.synsets(e[0][0][1][0])[0]
                print(e[0][0][1][0])
            else:
                s = wn.synsets(e[0][0][1])[0]


            similarity = s.wup_similarity(n)
            #print("similarity between {} and {} = {}".format(s,n, similarity))
            if similarity > wpthreshold:
                #print("words {} and {} is NOT a metaphor".format(adj,noun))
                return False


        #print("words {} and {} is a metaphor".format(adj,noun))
        return True

    else:
        #Word domain method

        wloader = wl.WordNetDomains(os.getcwd())
        nDomains = wloader.get_domains(noun)

        # if "factotum" in nDomains:
        #     nDomains.remove("factotum")
        #factotum in all words?

        for e in S1:
            if usingBNC:
                colDomains = wloader.get_domains(e[0][0][1][0])
            else:
                colDomains = wloader.get_domains(e[0][0][1])

            #print("domains in {} : {}".format(e[0][0][1], colDomains))
            #print("domains in {} : {}".format(noun, nDomains))


            for d in colDomains:
                if d in nDomains:
                    #print("words {} and {} is NOT a metaphor".format(adj,noun))
                    return False
        #print("words {} and {} is a metaphor".format(adj,noun))
        return True



def testCorpusTest(WuPalmer=True):
    print("testing with annotated corpus")
    mGuess = []

    for count, line in enumerate(lines):
        print("\r processing: " + str(count) + "/" + str(len(lines)), end="")

        tokens = line.split()

        del tokens[-1]
        #print(tokens)
        #print(type3metaphors)

        words = []
        for w in type3metaphors:
            if w[0][0] in tokens and w[0][1] in tokens:
                words = w[0]


        sent = ""
        for t in words:
            sent += " " + t

        cat = getWordCategories(sent)
        if cat[1] is not None:

            mGuess.append(str(count) + " : " + str(findIfMetaphor(cat, WuPalmer)))
        else:
            #print("no adjective")
            mGuess.append(str(count) + ": None")

    #calculate accuracy(= correct/all)
    #print(mGuess)
    numAll = 0
    numCorrect = 0
    falsePos = 0
    falseNeg = 0
    for count, gT in enumerate(groundTruth):
        if gT == "s" or "None" in mGuess[count] :
            continue
        else:
            numAll += 1
            predictedMetaphor = "True" in mGuess[count]
            isMetaphor = gT == "y"
            if predictedMetaphor == isMetaphor:
                numCorrect += 1
            elif predictedMetaphor:
                falsePos += 1
            elif not predictedMetaphor:
                falseNeg += 1
    accuracy = numCorrect/numAll


    print("\n Using threshold: " + str(wpthreshold))
    print("num All: " + str(numAll))
    print("num Correct: " + str(numCorrect))
    print("accuracy:" + str(accuracy))
    print("num false positives: " + str(falsePos))
    print("num false negatives: " +str(falseNeg))



def metaphorListTest(metaLines,type3Meta,doa,  WuPalmer=True):
    print("testing with metaphorList")
    mGuess = []
    for count, line in enumerate(metaLines):
        print("\r processing: " + str(count) + "/" + str(len(metaLines)), end="")

        tokens = line.split()

        del tokens[0]



        words = []
        for w in type3Meta:
            if w[0][0] in tokens and w[0][1] in tokens:
                words = w[0]
                #print(words)

        sent = ""
        for t in words:
            sent += " " + t


        cat = getWordCategories(sent)

        if cat[1] is not None:

            mGuess.append(str(count) + " : " + str(findIfMetaphor(cat, WuPalmer)))
        else:
            #print("no adjective")
            mGuess.append(str(count) + ": None")

    #calculate accuracy(= correct/all)
    numAll = 0
    numCorrect = 0
    falsePos = 0
    falseNeg = 0
    for count, gT in enumerate(doa):
        if "None" in mGuess[count] :
            continue
        else:
            numAll += 1
            predictedMetaphor = "True" in mGuess[count]
            isMetaphor = doa[count]
            if predictedMetaphor == isMetaphor:
                numCorrect += 1
            elif predictedMetaphor:
                falsePos += 1
            elif not predictedMetaphor:
                falseNeg += 1
    accuracy = numCorrect/numAll


    print("\n Using threshold: " + str(wpthreshold))
    print("num All: " + str(numAll))
    print("num Correct: " + str(numCorrect))
    print("accuracy:" + str(accuracy))
    print("num false positives: " + str(falsePos))
    print("num false negatives: " +str(falseNeg))





#1. Find metaphors for words in BNC

#create bigram finder
print("Creating bigram finder")
global_finder = BigramCollocationFinder.from_words(content)
print("Created bigrams")

miThresh = 5
print("Mi threshold = " + str(miThresh))
# print("Using mutual information method")
testWords = ["woman", "use", "dream", "body"]
mets = findMutualInformation(content, testWords, size_corpus, 4)
#print("found metaphors:")
#print(mets)

#2. test Metaphor finding w/ testCorpus
headwords = {}
groundTruth = []
f = open("testCorpus.txt", "r")
lines = f.readlines()
parseTestC() #initializes headwords and groundTruth
#print("accuracy: " + str(calAvgMi()))

#second way
testCorpusNS = [w for w in testCorpus if w not in es]
testFinder = BigramCollocationFinder.from_words(testCorpusNS)
punctfilter = lambda *w: len(w[0]) <= 1 or len(w[1]) <= 1
testFilter = lambda w1, w2: 'JJ' not in nltk.pos_tag([w1])[0][1] or "NN" not in nltk.pos_tag([w2])[0][1]
testFinder.apply_ngram_filter(punctfilter)
testFinder.apply_ngram_filter(testFilter)

type3metaphors = testFinder.ngram_fd.items()
#print(type3metaphors)






#3 & 4.Test Compatibility using wu and palmer
wpthreshold = 0.4

sen1 = "cold room"
sen2 = "He was a rather frightened flower in her presence"


# cats = getWordCategories(sen1)
# print("Sentence 1 (not metaphor): "+ str(findIfMetaphor(cats, True)))
#
# cats = getWordCategories(sen2)
# print("Sentence 2 (metaphor): "+ str(findIfMetaphor(cats, True)))


#5. Test with annotated corpus

#print("Using WuP")
#testCorpusTest(True)


#print("using WordNet")
#testCorpusTest(False)
# #6.Using wordnets
# print("Using WordNet Method to check word compatibility")
# cats = getWordCategories(sen1)
# print("Sentence 1 (not a metaphor): "+ str(findIfMetaphor(cats, False)))
#
# cats = getWordCategories(sen2)
# print("Sentence 2 (metaphor): "+ str(findIfMetaphor(cats, False)))
#
# #7. test with annotated corpus
#testCorpusTest(False)

#3rd list
#deadOrAlive, metList, metLines = rml.readMetList()
#metaphorListTest(metLines, metList, deadOrAlive, False)
