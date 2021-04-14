# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 12:18:36 2019

@author: P01004lr
"""

#KNN Test File
import re
import pandas as pd
from KNNTextClassifier import KNearestNeighbor
import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import genesis
from nltk.corpus import stopwords
genesis_ic = wn.ic(genesis, False, 0.0)
s = stopwords.words('english')
s.extend(['today', 'tomorrow', 'outside', 'out', 'there'])

import numpy as np

def main():
    FILENAME = "https://raw.githubusercontent.com/watson-developer-cloud/natural-language-classifier-nodejs/master/training/weather_data_train.csv"
    dataset = pd.read_csv(FILENAME, header = None)
    dataset.rename(columns = {0:'text', 1:'answer'}, inplace = True)
    dataset['output'] = np.where(dataset['answer'] == 'temperature', 1,0)

    print(dataset.head())
    print("\nSize of input file is ", dataset.shape)
    ps = nltk.wordnet.WordNetLemmatizer()
    for i in range(dataset.shape[0]):
        review = re.sub('[^a-zA-Z]', ' ', dataset.loc[i,'text'])
        review = review.lower()
        review = review.split()

    review = [ps.lemmatize(word) for word in review if not word in s]
    review = ' '.join(review)
    dataset.loc[i, 'text'] = review

    X_train = dataset['text']
    y_train = dataset['output']
    classifier = KNearestNeighbor(k=1, distance_type='path')
    classifier.train(X_train, y_train)

    final_test_list = ['will it rain', 'Is it hot outside?' , 'What is the expected high for today?',
                       'Will it be foggy tomorrow?', 'Should I prepare for sleet?',
                         'Will there be a storm today?', 'do we need to take umbrella today',
                        'will it be wet tomorrow', 'is it humid tomorrow', 'what is the precipitation today',
                        'is it freezing outside', 'is it cool outside', "are there strong winds outside",]
    test_corpus = []
    for i, value in enumerate(final_test_list):
        review = re.sub('[^a-zA-Z]', ' ', final_test_list[i])
        review = review.lower()
        review = review.split()
        review = [ps.lemmatize(word) for word in review if not word in s]
        review = ' '.join(review)
        test_corpus.append(review)
    y_pred_final = classifier.getPrediction(test_corpus)
    output_df = pd.DataFrame(data = {'text': final_test_list, 'code': y_pred_final})
    output_df['answer'] = np.where(output_df['code']==1, 'Temperature','Conditions')
    print(output_df)
if __name__ == '__main__':
     main()
