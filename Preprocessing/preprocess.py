from nltk import stem
from nltk.tokenize import wordpunct_tokenize
from sklearn.feature_extraction.text import CountVectorizer
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


def check_type(s):
	if type(s) != str:
		return s.encode('ascii', 'replace')
	return s


def remove_quotes(s):
	if not (s[0] == 'b' and (s[1] == '\'' or s[1] == '\"')):
		return s
	final = ""
	match = re.findall("b[\'\"].*?[\'\"] ", s)
	# while match is not None:
	for sentence in match:
		final += sentence[2:-2]
	return final


def get_sentiment(s):
	return sid.polarity_scores(s)


class Preprocess():
	def __init__(self, add_sentiment=False):
  		self.vectorizer = CountVectorizer(lowercase=True, stop_words=None, ngram_range=(2, 3))
		self.data_train, self.data_test, self.y_train, self.y_test = self.preprocess(add_sentiment)
		self._add_sentiment = add_sentiment

	def offset_data(self, n, pos, neg):
		offset_pos = pos[:]  # copy by value
		offset_neg = neg[:]
		label = "Top"
		if (n == int(n)):
			for ii in range(1, 21):
				curLabel = label + str(ii)
				offset_pos[curLabel] = pos[curLabel].shift(n)
				offset_neg[curLabel] = neg[curLabel].shift(n)
		return offset_pos, offset_neg

	def preprocess(self, add_sentiment):
		# Grab our data into pandas dataframe
		stemmer = stem.PorterStemmer()
		data = pd.read_csv('data/Combined_News_DJIA.csv')

		data_pos = data[data['Label'] == 1]
		data_neg = data[data['Label'] == 0]

		num_of_days = 1
		# offset headlines by n days
		data_pos, data_neg = self.offset_data(num_of_days, data_pos, data_neg)

		# Remove the stop words
		# stop = stopwords.words('english')
		# for line in data_pos:
		#	line = [l.lower() for l in wordpunct_tokenize(line) if l.lower() not in stop]

		# Put the data into a vectors
		data_pos_list = []
		data_neg_list = []
		for row in range(0, len(data_pos.index)):
			# if(row == 1):
			# 	print data_pos.iloc[row, 2:27]
			data_pos_list.append(' '.join(str(x) for x in data_pos.iloc[row, 2:27]))
		for row in range(0, len(data_neg.index)):
			data_neg_list.append(' '.join(str(x) for x in data_neg.iloc[row, 2:27]))

		# Stem and remove the b'' surrounding every headline
		# Also turns unicode into strings
		for row in range(0, len(data_pos_list)):
			data_pos_list[row] = stemmer.stem(re.sub('[^A-Za-z0-9.,\'\" ]+', '', data_pos_list[row]))
			data_pos_list[row] = check_type(data_pos_list[row])
			data_pos_list[row] = remove_quotes(data_pos_list[row])
		# sentiment = get_sentiment(data_pos_list[row])
		# data_pos_list[row] = [data_pos_list[row], sentiment['neg'], sentiment['pos']]

		for row in range(0, len(data_neg_list)):
			data_neg_list[row] = stemmer.stem(re.sub('[^A-Za-z0-9.,\'\" ]+', '', data_neg_list[row]))
			data_neg_list[row] = check_type(data_neg_list[row])
			data_neg_list[row] = remove_quotes(data_neg_list[row])
		# sentiment = get_sentiment(data_neg_list[row])
		# data_neg_list[row] = [data_neg_list[row], sentiment['neg'], sentiment['pos']]

		# random.shuffle(data_pos_list)
		# random.shuffle(data_neg_list)

		pos_split = int(len(data_pos_list) * .8)
		neg_split = int(len(data_neg_list) * .8)

		# Split the data
		data_train = data_pos_list[:pos_split] + data_neg_list[:neg_split]
		data_test = data_pos_list[pos_split:] + data_neg_list[neg_split:]
		y_train = np.append(np.ones((1, pos_split)), (np.zeros((1, neg_split))))
		y_test = np.append(np.ones((1, len(data_pos_list) - pos_split)), np.zeros((1, len(data_neg_list) - neg_split)))

		train_sent = []
		test_sent = []
		if (add_sentiment):
			for row in range(0, len(data_train)):
				sentiment = get_sentiment(data_train[row])
				train_sent.append([sentiment['neg'], sentiment['pos']])
			for row in range(0, len(data_test)):
				sentiment = get_sentiment(data_test[row])
				test_sent.append([sentiment['neg'], sentiment['pos']])

		# vectorizer = CountVectorizer(lowercase=True, stop_words=None, ngram_range=(2, 3))
		data_train = self.vectorizer.fit_transform(data_train)
		data_test = self.vectorizer.transform(data_test)

		if (add_sentiment):
			train_sent = csr_matrix(train_sent)
			test_sent = csr_matrix(test_sent)
			data_train = hstack((data_train, train_sent))
			data_test = hstack((data_test, test_sent))

		return data_train, data_test, y_train, y_test

	def get_results(self):
		return [self.data_train, self.data_test, self.y_train, self.y_test]

 	def transform_data(self, data):          
		 line = data
		 stemmer = stem.PorterStemmer()
		 stemmer.stem(re.sub('[^A-Za-z0-9.,\'\" ]+', '', line))
		 line = check_type(line)
		 line = remove_quotes(line)
		 result = self.vectorizer.transform([line])
		 
		 if(self._add_sentiment):
  			test_sent = []
  			sentiment = get_sentiment(line)
			test_sent.append([sentiment['neg'], sentiment['pos']])
			test_sent = csr_matrix(test_sent)
			result = hstack((result, test_sent))

		 return result

def main():
	preprocess = Preprocess()
	results = preprocess.get_results()
	pass


if __name__ == '__main__':
	main()
