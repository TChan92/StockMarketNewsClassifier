import pandas as pd

# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score
#
# ### Tutorial Code ###
# # Follow along here : https://www.kaggle.com/ndrewgele/omg-nlp-with-the-djia-and-reddit
#
# # Grab the data
# data = pd.read_csv('../data/Combined_WorldNews_DJIA.csv')
#
# # Split the data into test and training data
# train = data[data['Date'] < '2015-01-01']
# test = data[data['Date'] > '2015-12-31']
#
# # Grabs some data
# example = train.iloc[3,10]
# #print(example)
#
# # Transform all the words in the data to lower case
# example2 = example.lower()
# #print(example2)
#
# # Turn the data into a vector
# example3 = CountVectorizer().build_tokenizer()(example2)
# #print(example3)
#
# # Pretty print each word and how many times it appears in the vector
# #print pd.DataFrame([[x,example3.count(x)] for x in set(example3)], columns = ['Word', 'Count'])
#
# # Now that we have th functions covered, lets create the training headlines
# trainheadlines = []
# for row in range(0, len(train.index)):
#     trainheadlines.append(' '.join(str(x) for x in train.iloc[row,2:27]))
#
# # Wow! Our resulting table contains counts for 31,675 different words!
# basicvectorizer = CountVectorizer()
# basictrain = basicvectorizer.fit_transform(trainheadlines)
# #print(basictrain.shape)
#
# basicmodel = LogisticRegression()
# basicmodel = basicmodel.fit(basictrain, train["Label"])
# testheadlines = []
# for row in range(0,len(test.index)):
#     testheadlines.append(' '.join(str(x) for x in test.iloc[row,2:27]))
# basictest = basicvectorizer.transform(testheadlines)
# predictions = basicmodel.predict(basictest)
#
# # Using basic logistic regression we get (61 + 100) / (378) = 43% accuracy
# print pd.crosstab(test["Label"], predictions, rownames=["Actual"], colnames=["Predicted"])
#
# # We can print out the weight of each word like this
# basicwords = basicvectorizer.get_feature_names()
# basiccoeffs = basicmodel.coef_.tolist()[0]
# coeffdf = pd.DataFrame({'Word' : basicwords,
#                         'Coefficient' : basiccoeffs})
# coeffdf = coeffdf.sort_values(['Coefficient', 'Word'], ascending=[0, 1])
# #print coeffdf.head(10)
#
# # Now lets try to improve our accuracy using n-grams
# advancedvectorizer = CountVectorizer(ngram_range=(2,2))
# advancedtrain = advancedvectorizer.fit_transform(trainheadlines)
# advancedmodel = LogisticRegression()
# advancedmodel = advancedmodel.fit(advancedtrain, train["Label"])
# testheadlines = []
# for row in range(0,len(test.index)):
#     testheadlines.append(' '.join(str(x) for x in test.iloc[row,2:27]))
# advancedtest = advancedvectorizer.transform(testheadlines)
# advpredictions = advancedmodel.predict(advancedtest)
#
# # Using basic logistic regression with n-grams we get (66 + 147) / (378) = 56% accuracy
# testheadlines = []
# for row in range(0,len(test.index)):
#     testheadlines.append(' '.join(str(x) for x in test.iloc[row,2:27]))
# advancedtest = advancedvectorizer.transform(testheadlines)
# advpredictions = advancedmodel.predict(advancedtest)
# print pd.crosstab(test["Label"], advpredictions, rownames=["Actual"], colnames=["Predicted"])
WORLD_NEWS = '../data/Combined_WorldNews_DJIA.csv'
ECONOMICS = '../data/Combined_Economics_Saved.csv'
STOCKS = '../data/Combined_stocks_Saved.csv'
US_NEWS = '../data/Combined_news_Saved.csv'
TECH = '../data/Combined_technology_Saved.csv'
NY = '../data/Combined_NYT_Saved.csv'

data = pd.read_csv(NY, error_bad_lines=False)
train = data[data['Date'] < '2015-01-01']
data_pos = train[train['Label'] == 1]
data_neg = train[train['Label'] == 0]
train_pos = data_pos.shape[0]
train_neg = data_neg.shape[0]
print 'TRAIN'
print float(train_pos) / float(train_pos+train_neg)

test = data[data['Date'] > '2015-01-01']
data_pos = test[test['Label'] == 1]
data_neg = test[test['Label'] == 0]
test_pos = data_pos.shape[0]
test_neg = data_neg.shape[0]
print 'TEST'
print float(test_pos) / float(test_pos+test_neg)

print 'FINAL'
print float(train_pos + test_pos) / float(train_pos + test_pos + test_neg +train_neg)