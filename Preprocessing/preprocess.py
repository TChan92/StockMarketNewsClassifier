from nltk import stem
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd
import re

# Grab our data into pandas dataframe
stemmer = stem.PorterStemmer()
data = pd.read_csv('../data/Combined_News_DJIA.csv')
data_pos = data[data['Label'] == 1]
data_neg = data[data['Label'] == 0]


# Remove the stop words
stop = stopwords.words('english')
for line in data_pos:
	line = [l.lower() for l in wordpunct_tokenize(line) if l.lower() not in stop]

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

# pos_split = len(data_pos_list) * .9
# neg_split = len(data_neg_list) * .9

# Split the data into
# data_train = data_pos_list[:pos_split] + data_neg_list[:neg_split]
# data_test = data_pos_list[pos_split:] + data_neg_list[neg_split:]
# y_train = np.append(np.ones((1,pos_split)), (np.zeros((1,neg_split))))
# y_test = np.append(np.ones((1,len(data_pos_list) - pos_split)), np.zeros((1,len(data_neg_list) - neg_split)))

vectorizer = CountVectorizer(lowercase=True, stop_words='english',  max_df=1.0, min_df=1, max_features=None,  binary=True, ngram_range=(2,2))
pos_data = vectorizer.fit_transform(data_pos_list)
temp = vectorizer.inverse_transform(pos_data)
neg_data = vectorizer.transform(data_neg_list)

pass