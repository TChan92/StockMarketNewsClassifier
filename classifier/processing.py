import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from nltk import stem
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
import re

class Classifier():
	def __init__(self, params):
		self._train = params["train"]	# Train Data
		self._test = params["test"]  	# Test Data
		self._model = params["model"] 	# Classifier

		self._trainheadlines = []
		for row in range(0, len(self._train.index)):
			self._trainheadlines.append(' '.join(str(x) for x in self._train.iloc[row, 2:27]))

		vectorizer = CountVectorizer(ngram_range=(2, 3))
		advancedtrain = vectorizer.fit_transform(self._trainheadlines)
		self._model = self._model.fit(advancedtrain, self._train["Label"])

		self._testheadlines = []
		for row in range(0, len(self._test.index)):
			self._testheadlines.append(' '.join(str(x) for x in self._test.iloc[row, 2:27]))
		self._test = vectorizer.transform(self._testheadlines)
		advpredictions = self._model.predict(self._test)
		print (confusion_matrix(test["Label"], advpredictions))
		print(classification_report(test["Label"], advpredictions))

# Grab the data
data = pd.read_csv('../data/Combined_News_DJIA.csv')
stemmer = stem.PorterStemmer()
for row in range(0, len(data)):
	data[row] = stemmer.stem(re.sub('[^A-Za-z0-9.,\'\" ]+', '', data[row]))
train = data[data['Date'] < '2015-01-01']
test = data[data['Date'] > '2014-12-31']

params1 = {"train": train, "test": test, "model": LogisticRegression()}
lr = Classifier(params1)

params2 = {"train": train, "test": test, "model" : MLPClassifier(hidden_layer_sizes=(90, 80, 70))}
lr2 = Classifier(params2)
