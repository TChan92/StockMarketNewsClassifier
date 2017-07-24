from nltk import stem
from nltk.tokenize import wordpunct_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import deque
import pandas as pd
import re
import numpy as np
import random
from nltk.sentiment.util import *
from nltk.sentiment import sentiment_analyzer
from nltk.corpus import subjectivity
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from scipy.sparse import csr_matrix, coo_matrix, hstack

sid = SentimentIntensityAnalyzer()
tfid = TfidfVectorizer()


def check_type(s):
	if type(s) != str:
		return s.encode('ascii', 'replace')
	return s


# For relation
def get_row(row, data):
	if (row < 0 or row > len(data) - 1):
		return ""
	else:
		a = data[row].replace('nan', '')
		return a


def remove_quotes(s):
	if not (s[0] == 'b' and (s[1] == '\'' or s[1] == '\"')):
		return s
	final = ""
	match = re.findall("b[\'\"].*?[\'\"] ", s)
	# while match is not None:
	for sentence in match:
		final += sentence[2:-2]
	return final


class Preprocess():
	def __init__(self, data, offset, add_sentiment=True, stemming=True, add_date=False, transform_dates=False, add_relations = False):
		self.vectorizer = CountVectorizer(lowercase=True, stop_words=None, ngram_range=(2, 3))
		self.stemming = stemming
		self.offset = offset
		self.transform_dates = transform_dates
		self.add_date = add_date
		self.data = data
		self.add_sentiment = add_sentiment
		self.add_relation = add_relations
		self.data_train, self.data_test, self.y_train, self.y_test = self.preprocess()

	def get_sentiment(self, s):
		if (self.add_date):
			return sid.polarity_scores(s[0])
		else:
			return sid.polarity_scores(s)

	def offset_data(self, n, pos, neg, num_headlines):
		offset_pos = pos[:]  # copy by value
		offset_neg = neg[:]
		label = "Top"
		if (n == int(n)):
			for ii in range(1, num_headlines):
				curLabel = label + str(ii)
				offset_pos[curLabel] = pos[curLabel].shift(n)
				offset_neg[curLabel] = neg[curLabel].shift(n)
		return offset_pos, offset_neg

	def transform_date(self, date):
		date = date.split("/")
		return date[2] + "-" + date[0] + "-" + date[1]

	def extract_data_from_frame(self, data, num_headlines):
		data_list = []
		stemmer = stem.PorterStemmer()
		for row in range(0, len(data.index)):
			data_list.append(' '.join(str(x) for x in data.iloc[row, 2:num_headlines + 2]))
		for row in range(0, len(data_list)):
			if self.stemming:
				data_list[row] = stemmer.stem(re.sub('[^A-Za-z0-9.,\'\" ]+', '', data_list[row]))
				data_list[row] = check_type(data_list[row])
				data_list[row] = remove_quotes(data_list[row])
		if self.add_date:
			date_list = []
			for row in data['Date']:
				row = row.split('-')
				int_row = [int(a) for a in row]
				date_list.append(int_row)
			for row in range(0, len(data_list)):
				data_list[row] = [data_list[row]] + date_list[row]
		return data_list

	def vectorize_date(self, train, test):
		if self.add_date:
			words_train = [row[0] for row in train]
			words_test = [row[0] for row in test]
			dates_train = csr_matrix([row[-3:] for row in train])
			dates_test = csr_matrix([row[-3:] for row in test])
			data_train = hstack((self.vectorizer.fit_transform(words_train), dates_train))
			data_test = hstack((self.vectorizer.transform(words_test), dates_test))
		else:
			data_train = self.vectorizer.fit_transform(train)
			data_test = self.vectorizer.transform(test)
		return data_train, data_test

	def preprocess(self):
		# Grab our data into pandas dataframe
		data = pd.read_csv(self.data, error_bad_lines=False)
		num_headlines = data.shape[1] - 2

		if self.transform_dates:
			data['Date'] = data['Date'].map(lambda a: self.transform_date(a))
		data_pos = data[data['Label'] == 1]
		data_neg = data[data['Label'] == 0]

		num_of_days = self.offset
		# offset headlines by n days
		data_pos, data_neg = self.offset_data(num_of_days, data_pos, data_neg, num_headlines)

		pos_split = len(data_pos[data_pos['Date'] < '2015-01-01'])
		neg_split = len(data_neg[data_neg['Date'] < '2015-01-01'])

		# Put the data into a vectors
		# Stem and remove the b'' surrounding every headline
		# Also turns unicode into strings
		data_pos_list = self.extract_data_from_frame(data_pos, num_headlines)
		data_neg_list = self.extract_data_from_frame(data_neg, num_headlines)

		# Split the data
		data_train = data_pos_list[:pos_split] + data_neg_list[:neg_split]
		data_test = data_pos_list[pos_split:] + data_neg_list[neg_split:]
		y_train = np.append(np.ones((1, pos_split)), (np.zeros((1, neg_split))))
		y_test = np.append(np.ones((1, len(data_pos_list) - pos_split)), np.zeros((1, len(data_neg_list) - neg_split)))

		train_sent = []
		test_sent = []
		if (self.add_sentiment):
			for row in range(0, len(data_train)):
				sentiment = self.get_sentiment(data_train[row])
				train_sent.append([sentiment['neg'], sentiment['pos']])
			for row in range(0, len(data_test)):
				sentiment = self.get_sentiment(data_test[row])
				test_sent.append([sentiment['neg'], sentiment['pos']])

		train_relation_metric = []
		test_relation_metric = []
		if (self.add_relation):
			for row in range(0, len(data_train)):
				relation = self.get_relation(row, data_train)
				train_relation_metric.append(relation)
			for row in range(0, len(data_test)):
				relation = self.get_relation(row, data_test)
				test_relation_metric.append(relation)

		# vectorizer = CountVectorizer(lowercase=True, stop_words=None, ngram_range=(2, 3))
		data_train, data_test = self.vectorize_date(data_train, data_test)

		if (self.add_sentiment):
			train_sent = csr_matrix(train_sent)
			test_sent = csr_matrix(test_sent)
			data_train = hstack((data_train, train_sent))
			data_test = hstack((data_test, test_sent))
		if (self.add_relation):
			train_relation_metric = csr_matrix(train_relation_metric)
			test_relation_metric = csr_matrix(test_relation_metric)
			data_train = hstack((data_train, train_relation_metric))
			data_test = hstack((data_test, test_relation_metric))

		return data_train, data_test, y_train, y_test

	def get_results(self):
		return [self.data_train, self.data_test, self.y_train, self.y_test]

	def get_relation(self, row, data):
		if self.add_date:
			data = [a[0] for a in data]
		r = [get_row(row - 2, data), get_row(row - 1, data), data[row], get_row(row + 1, data), get_row(row + 2, data)]
		t = tfid.fit_transform(r)
		sims = 0
		for a in [0, 1, 3, 4]:
			sims += cosine_similarity(t[a], t[2])[0]
		pass
		return sims / 4


def main():
	preprocess = Preprocess()
	results = preprocess.get_results()
	pass


if __name__ == '__main__':
	main()
