import nltk
from nltk.collocations import *
from nltk.corpus import PlaintextCorpusReader
import numpy as np
import os





#import brown corpus
from nltk.corpus import brown

#import testC
path = os.path.dirname(os.path.abspath(__file__))
files = ".*\.txt"
corpus0 = PlaintextCorpusReader(path, files)
testCorpus  = nltk.Text(corpus0.words())
