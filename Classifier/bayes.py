import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt
from nltk.corpus import treebank
from nltk import stem
import re
import math as m
import pandas as pd

# Grab our data into pandas dataframe
stemmer = stem.PorterStemmer()
data = pd.read_csv('../data/Combined_News_DJIA.csv')
data_pos = data[data['Label'] == 1]
data_neg = data[data['Label'] == 0]

# Put the data into a vectors
data_pos_list = []
data_neg_list = []
for row in range(0, len(data_pos.index)):
	data_pos_list.append(' '.join(str(x) for x in data_pos.iloc[row,2:27]))
for row in range(0, len(data_neg.index)):
	data_neg_list.append(' '.join(str(x) for x in data_neg.iloc[row,2:27]))

for row in range(0, len(data_pos_list)):
	data_pos_list[row] = stemmer.stem(re.sub('[^A-Za-z0-9., ]+', '', data_pos_list[row]))
for row in range(0, len(data_neg_list)):
	data_neg_list[row] = stemmer.stem(re.sub('[^A-Za-z0-9., ]+', '', data_neg_list[row]))


data_train = data_pos_list[0:852] + data_neg_list[0:740]
data_test = data_pos_list[852:] + data_neg_list[740:]
y_train = np.append(np.ones((1,852)), (np.zeros((1,740))))
y_test = np.append(np.ones((1,213)), np.zeros((1,184)))

# Process the data
vectorizer = CountVectorizer(lowercase=True, stop_words='english',  max_df=1.0, min_df=1, max_features=None,  binary=True, ngram_range=(2,2))
X_train = vectorizer.fit_transform(data_train)
X_test = vectorizer.transform(data_test)

for a in range(1, 30):
	alpha = float(a)/10
	model = MultinomialNB(alpha=a, class_prior=None, fit_prior=True)
	model.fit(X_train, y_train)
	correct = 0.0
	for i in range(0, len(y_test)):
		if model.predict(X_test[i])[0] == y_test[i]:
			correct += 1.0
	print (a,correct/len(y_test))
