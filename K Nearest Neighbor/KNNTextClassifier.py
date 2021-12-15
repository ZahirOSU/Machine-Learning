# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 09:55:10 2019

"""
import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import genesis
from nltk.corpus import stopwords

genesis_ic = wn.ic(genesis, False, 0.0)
s = stopwords.words('english')
s.extend(['today', 'tomorrow', 'outside', 'out', 'there'])

import numpy as np
from nltk.tokenize import word_tokenize

class KNearestNeighbor:
    #Constructor
    def __init__(self, k=1, distance_type = 'path'):
        self.k = k
        self.distance_type = distance_type

    #Assigning training data to class variables
    def train(self, train1, train2):
        self.train1 = train1
        self.train2 = train2
    #Predicting data. For every row of text data, compare the test data with the training data.
    #We get a similarity score. Runs in O(m*n) time where m = no. of rows train and n = no. of rows test
    def getPrediction(self, test):
        self.test = test
        prediction = []

        for i, value in enumerate(test):
            max_similar = 0
            max_index = 0
            for j in range(self.train1.shape[0]):
                temp = self.document_similarity(test[i], self.train1[j])
                if temp > max_similar:
                    max_similar = temp
                    max_index = j
            prediction.append(self.train2[max_index])
        return prediction

    #POS tag conversion from nltk to wordnet.synsets
    @staticmethod
    def convert_tag(tag):
        tag_dict = {'N':'n', 'J': 'a', 'R':'r', 'V':'v'}
        try:
            return tag_dict[tag[0]]
        except KeyError:
            return None

    #This is the function that returns the synsets and converts the documents.
    def doc_to_synsets(self, doc):
        tokens = word_tokenize(doc+' ')
        l=[]
        tags = nltk.pos_tag([tokens[0] + ' ']) if len(tokens) == 1 else nltk.pos_tag(tokens)

        for token, tag in zip(tokens, tags):
            syntag = self.convert_tag(tag[1])
            syns = wn.synsets(token, syntag)
            if (len(syns) > 0):
                l.append(syns[0])
        return l

    #Gets document similarity number. 0 means low similarity and 1 means high similarity
    def document_similarity(self,doc1, doc2):

          synsets1 = self.doc_to_synsets(doc1)
          synsets2 = self.doc_to_synsets(doc2)

          return (self.similarity(synsets1, synsets2) + self.similarity(synsets2, synsets1)) / 2
    #Gets the similarity score between two synsets(instances of data)
    @staticmethod
    def similarity(s1, s2, distance_type = 'path'):
        s1_largest_scores = []
        for i, s1_synset in enumerate(s1, 0):
            max_score = 0
            for s2_synset in s2:
                if distance_type == 'path':
                    score = s1_synset.path_similarity(s2_synset, simulate_root= False)
                else:
                    score = s1_synset.wup_similarity(s2_synset)
                if score != None:
                    if score > max_score:
                        max_score = score
            if max_score != 0:
                s1_largest_scores.append(max_score)
        mean_score = np.mean(s1_largest_scores)
        return mean_score
