import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
from sklearn import svm

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

	def predict(self):
		advpredictions = self._model.predict(self._test)
		print pd.crosstab(test["Label"], advpredictions, rownames=["Actual"], colnames=["Predicted"])


# Grab the data
data = pd.read_csv('../data/Combined_News_DJIA.csv')
train = data[data['Date'] < '2015-01-01']
test = data[data['Date'] > '2014-12-31']

params1 = {"train": train, "test": test, "model": LogisticRegression()}
lr = Classifier(params1)
lr.predict()

params2 = {"train": train, "test": test, "model": svm.SVR()}
lr = Classifier(params2)
lr.predict()