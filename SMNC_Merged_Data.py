from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.linear_model import *
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def get_valid_dates(d1, d2, d3):
	d1_train, d1_test = get_dates(d1, False)
	d2_train, d2_test = get_dates(d2, True)
	d3_train, d3_test = get_dates(d3, False)
	train_dates = list(set(d1_train) & set(d2_train) & set(d3_train))
	test_dates = list(set(d1_test) & set(d2_test) & set(d3_test))
	print len(train_dates)
	print len(test_dates)


def transform_date(date):
	date = date.split("/")
	return date[2] + "-" + date[0] + "-" + date[1]


def get_dates(data_set, change_format):
	if change_format:
		data_set['Date'] = data_set['Date'].map(lambda a: transform_date(a))
	train = data_set[data_set['Date'] < '2015-11-30']
	test = data_set[data_set['Date'] > '2015-12-01']
	train_headlines, test_headlines = [], []
	for row in range(0, len(train.index)):
		train_headlines.append(train.iloc[row, 0])
	for row in range(0, len(test.index)):
		test_headlines.append(test.iloc[row, 0])
	return train_headlines, test_headlines


AGGREGATED_TRAIN = []
AGGREGATED_TEST = []

# Grab the wn data
data_wn = pd.read_csv('data/Combined_News_DJIA.csv')
data_e = pd.read_csv('data/Combined_Economics_Saved.csv')
data_stocks = pd.read_csv('data/Combined_stocks_Saved.csv')


valid_headlines_train = []
valid_headlines_test = []
# STOCKS
train_stocks = data_stocks[data_stocks['Date'] < '2015-11-30']
test_stocks = data_stocks[data_stocks['Date'] > '2015-12-01']

train_headlines_stocks = []
for row in range(0, len(train_stocks.index)):
	valid_headlines_train.append(train_stocks.iloc[row, 0])
	train_headlines_stocks.append(' '.join(str(x) for x in train_stocks.iloc[row, 2:12]))

vectorizer = CountVectorizer(lowercase=True, stop_words=None, ngram_range=(2, 3))
train_headlines_stocks = vectorizer.fit_transform(train_headlines_stocks)
clf = SGDClassifier(shuffle=False, n_iter=1000, loss='squared_hinge')

clf = clf.fit(train_headlines_stocks, train_stocks["Label"])
test_headlines_stocks = []
for row in range(0, len(test_stocks.index)):
	valid_headlines_test.append(test_stocks.iloc[row, 0])
	test_headlines_stocks.append(' '.join(str(x) for x in test_stocks.iloc[row, 2:12]))
test_headlines_stocks = vectorizer.transform(test_headlines_stocks)

AGGREGATED_TRAIN = clf.predict(train_headlines_stocks)
AGGREGATED_TEST = clf.predict(test_headlines_stocks)

''' ------------------------------------------------------------ '''

# WN
train_wn = data_wn[data_wn['Date'] < '2015-11-30']
test_wn = data_wn[data_wn['Date'] > '2015-12-01']

train_headlines_wn = []
for row in range(0, len(train_wn.index)):
	if (train_wn.iloc[row, 0] in valid_headlines_train):
		train_headlines_wn.append(' '.join(str(x) for x in train_wn.iloc[row, 2:27]))

vectorizer = CountVectorizer(lowercase=True, stop_words=None, ngram_range=(2, 3))
train_headlines_wn = vectorizer.fit_transform(train_headlines_wn)
clf = SGDClassifier(shuffle=False, n_iter=1000, loss='squared_hinge')

clf = clf.fit(train_headlines_wn, train_stocks["Label"])
test_headlines_wn = []
for row in range(0, len(test_wn.index)):
	if (test_wn.iloc[row, 0] in valid_headlines_test):
		test_headlines_wn.append(' '.join(str(x) for x in test_wn.iloc[row, 2:27]))
test_headlines_wn = vectorizer.transform(test_headlines_wn)

AGGREGATED_TRAIN = zip(AGGREGATED_TRAIN, clf.predict(train_headlines_wn))
AGGREGATED_TEST = zip(AGGREGATED_TEST, clf.predict(test_headlines_wn))


clf = SGDClassifier(shuffle=False, n_iter=1000, loss='squared_hinge')
clf = clf.fit(AGGREGATED_TRAIN, train_stocks["Label"])
predictions = clf.predict(AGGREGATED_TEST)
print pd.crosstab(test_stocks["Label"], predictions, rownames=["Actual"], colnames=["Predicted"])
