import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle

# Grab the data
data = pd.read_csv('../data/Combined_News_DJIA.csv')
data_pos = data[data['Label'] == 1]
data_neg = data[data['Label'] == 0]

data_neg = shuffle(data_neg)
data_pos = shuffle(data_pos)

# Split the data into test and training data
train = pd.concat([data_pos[0:852], data_neg[0:740]])
test = pd.concat([data_pos[852:], data_neg[740:]])

# Now that we have th functions covered, lets create the training headlines
trainheadlines = []
for row in range(0, len(train.index)):
    trainheadlines.append(' '.join(str(x) for x in train.iloc[row,2:27]))

# Now lets try to improve our accuracy using n-grams
advancedvectorizer = CountVectorizer(ngram_range=(2, 3))
advancedtrain = advancedvectorizer.fit_transform(trainheadlines)
advancedmodel = LogisticRegression()
advancedmodel = advancedmodel.fit(advancedtrain, train["Label"])


# Using basic logistic regression with n-grams we get (66 + 147) / (378) = 56% accuracy
testheadlines = []
for row in range(0,len(test.index)):
    testheadlines.append(' '.join(str(x) for x in test.iloc[row,2:27]))
advancedtest = advancedvectorizer.transform(testheadlines)
advpredictions = advancedmodel.predict(advancedtest)
print pd.crosstab(test["Label"], advpredictions, rownames=["Actual"], colnames=["Predicted"])